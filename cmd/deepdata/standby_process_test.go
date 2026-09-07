package main

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"
)

// standbyProcessNodeToken is the node credential shared by every real process
// this test starts: DEEPDATA_LEADER_URL requires it to equal the leader's own
// DEEPDATA_REPLICATION_TOKEN, since a standby is also the leader's client.
const standbyProcessNodeToken = "standby-process-test-node-token-strong-credential"

var standbyProcessDocsSchema = []byte(`{"name":"docs","fields":[{"name":"embedding","type":"dense","dim":2,"index":{"type":"flat"}}]}`)

func standbyProcessDoc(id int, x, y float64) []byte {
	body, _ := json.Marshal(map[string]any{
		"id":      id,
		"vectors": map[string]any{"embedding": []float64{x, y}},
	})
	return body
}

func standbyProcessDocsURL(address, tenant, collection string) string {
	return "http://" + address + "/v3/tenants/" + tenant + "/collections/" + collection + "/docs"
}

// readyzJSON fetches and decodes /readyz on address. It never fails the test
// on a non-200: callers that care about the status check it themselves,
// since a caller polling through a leader outage expects 200 throughout.
func readyzJSON(t *testing.T, address string) (int, map[string]any) {
	t.Helper()
	resp, err := (&http.Client{Timeout: 2 * time.Second}).Get("http://" + address + "/readyz")
	if err != nil {
		t.Fatalf("GET %s/readyz: %v", address, err)
	}
	defer resp.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode %s/readyz body: %v", address, err)
	}
	return resp.StatusCode, body
}

// followingTenantState digs following.tenants.<tenant>.state out of a
// /readyz body, or "" if the standby has not reported that tenant yet.
func followingTenantState(body map[string]any, tenant string) string {
	following, _ := body["following"].(map[string]any)
	tenants, _ := following["tenants"].(map[string]any)
	entry, _ := tenants[tenant].(map[string]any)
	state, _ := entry["state"].(string)
	return state
}

// waitForFollowingState polls address's /readyz until tenant reports state,
// failing immediately on a non-200 response (a standby must stay ready while
// it follows) or after deadline if state is never reached.
func waitForFollowingState(t *testing.T, address, tenant, state string, deadline time.Duration) map[string]any {
	t.Helper()
	end := time.Now().Add(deadline)
	var last map[string]any
	for time.Now().Before(end) {
		status, body := readyzJSON(t, address)
		if status != http.StatusOK {
			t.Fatalf("GET %s/readyz = %d while waiting for tenant %q state %q: %+v", address, status, tenant, state, body)
		}
		last = body
		if followingTenantState(body, tenant) == state {
			return body
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("tenant %q on %s did not reach state %q within %s (last readyz: %+v)", tenant, address, state, deadline, last)
	return nil
}

// waitForDocReadable polls a GET on url until it returns 200 or deadline
// passes -- the shape of "an insert on the leader is readable on the standby
// shortly after".
func waitForDocReadable(t *testing.T, url string, deadline time.Duration) {
	t.Helper()
	end := time.Now().Add(deadline)
	var last *http.Response
	var lastBody []byte
	for time.Now().Before(end) {
		last, lastBody = canonicalJSONRequest(t, http.MethodGet, url, nil)
		if last.StatusCode == http.StatusOK {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("GET %s never returned 200 within %s (last: %d %s)", url, deadline, last.StatusCode, lastBody)
}

// TestStandbyProcessServesReadsWhileFollowing proves the whole standby loop
// against two real `deepdata serve` processes: a leader with the node
// surface enabled, and a second process pointed at it via
// DEEPDATA_LEADER_URL that follows every tenant the leader lists while
// answering this process's own reads.
//
// It does not exercise a tenant created on the leader after the standby
// started: replicateAll's re-list runs every relistInterval (30s), a
// package-level var this test's separate OS process cannot shorten from the
// outside. That path is covered in-process, with relistInterval turned down,
// by TestServeFollowServesReadsWhileFollowing in standby_test.go.
func TestStandbyProcessServesReadsWhileFollowing(t *testing.T) {
	leaderDir := filepath.Join(t.TempDir(), "leader")
	leaderHTTP := unusedLoopbackAddress(t)
	leaderGRPC := unusedLoopbackAddress(t)
	leader := startCanonicalTestProcess(t, leaderDir, leaderHTTP, leaderGRPC, map[string]string{
		"DEEPDATA_REPLICATION_TOKEN": standbyProcessNodeToken,
	})
	defer leader.stopIfRunning()
	leader.waitReady(t, leaderHTTP)

	acmeCollectionsURL := "http://" + leaderHTTP + "/v3/tenants/acme/collections"
	if resp, body := canonicalJSONRequest(t, http.MethodPost, acmeCollectionsURL, standbyProcessDocsSchema); resp.StatusCode != http.StatusCreated {
		t.Fatalf("create acme/docs on leader = %d: %s", resp.StatusCode, body)
	}
	for id := 1; id <= 3; id++ {
		if resp, body := canonicalJSONRequest(t, http.MethodPost, standbyProcessDocsURL(leaderHTTP, "acme", "docs"), standbyProcessDoc(id, 1, 0)); resp.StatusCode != http.StatusOK {
			t.Fatalf("insert doc %d on leader = %d: %s", id, resp.StatusCode, body)
		}
	}

	standbyDir := filepath.Join(t.TempDir(), "standby")
	standbyHTTP := unusedLoopbackAddress(t)
	standbyGRPC := unusedLoopbackAddress(t)
	standby := startCanonicalTestProcess(t, standbyDir, standbyHTTP, standbyGRPC, map[string]string{
		"DEEPDATA_LEADER_URL":        "http://" + leaderHTTP,
		"DEEPDATA_REPLICATION_TOKEN": standbyProcessNodeToken,
	})
	defer standby.stopIfRunning()
	standby.waitReady(t, standbyHTTP)
	waitForFollowingState(t, standbyHTTP, "acme", "streaming", 30*time.Second)

	for id := 1; id <= 3; id++ {
		url := standbyProcessDocsURL(standbyHTTP, "acme", "docs") + "/" + strconv.Itoa(id)
		if resp, body := canonicalJSONRequest(t, http.MethodGet, url, nil); resp.StatusCode != http.StatusOK {
			t.Fatalf("get doc %d on standby = %d: %s", id, resp.StatusCode, body)
		}
	}

	// An insert on the leader is readable on the standby within 5s.
	if resp, body := canonicalJSONRequest(t, http.MethodPost, standbyProcessDocsURL(leaderHTTP, "acme", "docs"), standbyProcessDoc(4, 0, 1)); resp.StatusCode != http.StatusOK {
		t.Fatalf("insert doc 4 on leader = %d: %s", resp.StatusCode, body)
	}
	waitForDocReadable(t, standbyProcessDocsURL(standbyHTTP, "acme", "docs")+"/4", 5*time.Second)

	// A write through the standby is refused: it is a read replica.
	if resp, body := canonicalJSONRequest(t, http.MethodPost, standbyProcessDocsURL(standbyHTTP, "acme", "docs"), standbyProcessDoc(0, 0, 0)); resp.StatusCode != http.StatusForbidden {
		t.Fatalf("write to standby = %d, want 403: %s", resp.StatusCode, body)
	}

	if err := leader.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	// ponytail: cmd/deepdata/main.go's gracefulShutdown gives http.Server.Shutdown
	// its own 30s context before forcing an active connection closed, and the
	// standby's follow stream is exactly such a connection (it is still
	// "active" from the leader's perspective, blocked reading the next
	// journal record, not idle) -- so the leader does not actually release the
	// socket, and the standby does not see the drop, until close to that 30s
	// mark. Upgrade path: give the node surface its own shorter shutdown
	// deadline, or close its connections eagerly when the server starts
	// shutting down, if operators need this to be faster than that.
	waitForFollowingState(t, standbyHTTP, "acme", "reconnecting", 45*time.Second)

	select {
	case <-leader.done:
		if err := leader.waitError(); err != nil {
			t.Fatalf("leader failed graceful shutdown: %v\n%s", err, leader.output.String())
		}
	case <-time.After(15 * time.Second):
		t.Fatalf("leader did not exit after SIGTERM\n%s", leader.output.String())
	}

	restartedLeader := startCanonicalTestProcess(t, leaderDir, leaderHTTP, leaderGRPC, map[string]string{
		"DEEPDATA_REPLICATION_TOKEN": standbyProcessNodeToken,
	})
	defer restartedLeader.stopIfRunning()
	restartedLeader.waitReady(t, leaderHTTP)

	if resp, body := canonicalJSONRequest(t, http.MethodPost, standbyProcessDocsURL(leaderHTTP, "acme", "docs"), standbyProcessDoc(5, 1, 1)); resp.StatusCode != http.StatusOK {
		t.Fatalf("insert doc 5 on restarted leader = %d: %s", resp.StatusCode, body)
	}
	// standbyRetry (5s, unexported, no env override) bounds one reconnect
	// attempt; give it room for that plus the bind.
	waitForFollowingState(t, standbyHTTP, "acme", "streaming", 15*time.Second)
	waitForDocReadable(t, standbyProcessDocsURL(standbyHTTP, "acme", "docs")+"/5", 5*time.Second)

	standby.terminate(t)

	tenantsDir := filepath.Join(standbyDir, "index.gob.tenants")
	for _, marker := range []string{"acme-replica", "acme.initialized"} {
		if _, err := os.Stat(filepath.Join(tenantsDir, marker)); err != nil {
			t.Errorf("standby tenants dir missing %s: %v", marker, err)
		}
	}

	plainHTTP := unusedLoopbackAddress(t)
	plainGRPC := unusedLoopbackAddress(t)
	plain := startCanonicalTestProcess(t, standbyDir, plainHTTP, plainGRPC)
	defer plain.stopIfRunning()
	plain.waitReady(t, plainHTTP)
	if status, body := readyzJSON(t, plainHTTP); status != http.StatusOK || body["read_only"] != true {
		t.Fatalf("plain serve on the former standby dir readyz = %d read_only=%v, want 200 true: %+v", status, body["read_only"], body)
	}
	plain.terminate(t)
}
