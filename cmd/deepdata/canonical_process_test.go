package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

const (
	canonicalProcessHelperEnv = "DEEPDATA_CANONICAL_PROCESS_HELPER"
	canonicalProcessAPIToken  = "canonical-process-test-token-strong-credential"
)

func TestCanonicalServerProcessHelper(t *testing.T) {
	if os.Getenv(canonicalProcessHelperEnv) != "1" {
		return
	}
	os.Args = []string{"deepdata", "serve"}
	main()
}

type canonicalTestProcess struct {
	cmd    *exec.Cmd
	output *bytes.Buffer
	done   chan struct{}
	mu     sync.Mutex
	err    error
}

func processEnv(overrides map[string]string) []string {
	result := make([]string, 0, len(os.Environ())+len(overrides))
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if _, replaced := overrides[key]; !replaced {
			result = append(result, entry)
		}
	}
	keys := make([]string, 0, len(overrides))
	for key := range overrides {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result = append(result, key+"="+overrides[key])
	}
	return result
}

func portFromAddress(t *testing.T, address string) string {
	t.Helper()
	_, port, err := net.SplitHostPort(address)
	if err != nil {
		t.Fatal(err)
	}
	return port
}

func startCanonicalTestProcess(
	t *testing.T,
	dataDir, httpAddress, grpcAddress string,
	extraOverrides ...map[string]string,
) *canonicalTestProcess {
	t.Helper()
	output := new(bytes.Buffer)
	cmd := exec.Command(os.Args[0], "-test.run=^TestCanonicalServerProcessHelper$")
	overrides := map[string]string{
		canonicalProcessHelperEnv:        "1",
		"API_TOKEN":                      canonicalProcessAPIToken,
		"COMPACT_INTERVAL_MIN":           "60",
		"DEEPDATA_INSECURE_DEV_MODE":     "",
		"DEEPDATA_ENABLE_LEGACY_RUNTIME": "",
		"GRPC_PORT":                      portFromAddress(t, grpcAddress),
		"HTTP_H2C":                       "0",
		"JWT_SECRET":                     "",
		"LOG_FORMAT":                     "text",
		"LOG_LEVEL":                      "error",
		"PORT":                           portFromAddress(t, httpAddress),
		"REQUIRE_AUTH":                   "1",
		"VECTORDB_BASE_DIR":              filepath.Dir(dataDir),
		"VECTORDB_DATA_DIR":              dataDir,
		"VECTORDB_MODE":                  "local",
	}
	for _, extra := range extraOverrides {
		for key, value := range extra {
			overrides[key] = value
		}
	}
	cmd.Env = processEnv(overrides)
	cmd.Stdout = output
	cmd.Stderr = output
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	process := &canonicalTestProcess{cmd: cmd, output: output, done: make(chan struct{})}
	go func() {
		err := cmd.Wait()
		process.mu.Lock()
		process.err = err
		process.mu.Unlock()
		close(process.done)
	}()
	return process
}

func TestCanonicalStartupRequiresCredentialUnlessExplicitlyInsecure(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	process := startCanonicalTestProcess(
		t,
		dataDir,
		unusedLoopbackAddress(t),
		unusedLoopbackAddress(t),
		map[string]string{
			"API_TOKEN":                  "",
			"JWT_SECRET":                 "",
			"REQUIRE_AUTH":               "0",
			"DEEPDATA_INSECURE_DEV_MODE": "",
		},
	)
	defer process.stopIfRunning()
	select {
	case <-process.done:
		if err := process.waitError(); err == nil {
			t.Fatalf("credentialless production helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("credentialless production helper did not exit\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "API_TOKEN or JWT_SECRET is required") {
		t.Fatalf("unexpected credentialless startup error:\n%s", process.output.String())
	}
	if _, err := os.Stat(dataDir); !os.IsNotExist(err) {
		t.Fatalf("credentialless startup touched data directory: err=%v", err)
	}
}

func TestCanonicalStartupAllowsExplicitInsecureDevelopmentMode(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	httpAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(
		t,
		dataDir,
		httpAddress,
		unusedLoopbackAddress(t),
		map[string]string{
			"API_TOKEN":                  "",
			"JWT_SECRET":                 "",
			"REQUIRE_AUTH":               "0",
			"DEEPDATA_INSECURE_DEV_MODE": "1",
		},
	)
	defer process.stopIfRunning()
	process.waitReady(t, httpAddress)
	process.terminate(t)
}

func TestValidateCanonicalAuthEnvironmentRejectsUnsafeCredentialModes(t *testing.T) {
	t.Setenv("DEEPDATA_INSECURE_DEV_MODE", "")
	strongAPI := strings.Repeat("a", canonicalCredentialMinBytes)
	strongJWT := strings.Repeat("j", canonicalCredentialMinBytes)
	for _, tc := range []struct {
		name       string
		apiToken   string
		jwtSecret  string
		wantErr    bool
		insecureOK string
	}{
		{name: "strong static", apiToken: strongAPI, wantErr: false},
		{name: "strong jwt", jwtSecret: strongJWT, wantErr: false},
		{name: "both", apiToken: strongAPI, jwtSecret: strongJWT, wantErr: true},
		{name: "short static", apiToken: "static", wantErr: true},
		{name: "short jwt", jwtSecret: "jwt", wantErr: true},
		{name: "static leading whitespace", apiToken: " " + strongAPI, wantErr: true},
		{name: "jwt trailing whitespace", jwtSecret: strongJWT + "\t", wantErr: true},
		{name: "neither", wantErr: true},
		{name: "explicit insecure development", insecureOK: "1", wantErr: false},
		{name: "insecure development still rejects configured weak credential", apiToken: "short", insecureOK: "1", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("API_TOKEN", tc.apiToken)
			t.Setenv("JWT_SECRET", tc.jwtSecret)
			t.Setenv("DEEPDATA_INSECURE_DEV_MODE", tc.insecureOK)
			err := validateCanonicalAuthEnvironment()
			if (err != nil) != tc.wantErr {
				t.Fatalf("validation error = %v, wantErr=%v", err, tc.wantErr)
			}
		})
	}
}

func TestCanonicalStartupRejectsWeakCredentialBeforeStateInitialization(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	process := startCanonicalTestProcess(
		t,
		dataDir,
		unusedLoopbackAddress(t),
		unusedLoopbackAddress(t),
		map[string]string{"API_TOKEN": "short", "JWT_SECRET": ""},
	)
	defer process.stopIfRunning()
	select {
	case <-process.done:
		if err := process.waitError(); err == nil {
			t.Fatalf("weak-credential helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("weak-credential helper did not exit\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "API_TOKEN must be at least 32 bytes") {
		t.Fatalf("unexpected weak-credential startup error:\n%s", process.output.String())
	}
	if _, err := os.Stat(dataDir); !os.IsNotExist(err) {
		t.Fatalf("weak-credential startup touched data directory: err=%v", err)
	}
}

func TestCanonicalStartupIgnoresInvalidLegacyCompactionInterval(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	httpAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(
		t,
		dataDir,
		httpAddress,
		unusedLoopbackAddress(t),
		map[string]string{"COMPACT_INTERVAL_MIN": "0"},
	)
	defer process.stopIfRunning()
	process.waitReady(t, httpAddress)
	process.terminate(t)
}

func TestCanonicalProductionProcessCannotEnableLegacyRuntime(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	httpAddress := unusedLoopbackAddress(t)
	grpcAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(
		t,
		dataDir,
		httpAddress,
		grpcAddress,
		map[string]string{"DEEPDATA_ENABLE_LEGACY_RUNTIME": "1"},
	)
	defer process.stopIfRunning()
	process.waitReady(t, httpAddress)

	for _, path := range []string{"/v2/collections", "/api/embed", "/api/config/embedder"} {
		response, body := canonicalJSONRequest(
			t,
			http.MethodGet,
			"http://"+httpAddress+path,
			nil,
		)
		if response.StatusCode != http.StatusNotFound {
			process.terminate(t)
			t.Fatalf("legacy opt-in exposed %s with status %d: %s", path, response.StatusCode, body)
		}
	}
	process.terminate(t)
}

func (p *canonicalTestProcess) waitError() error {
	<-p.done
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.err
}

func (p *canonicalTestProcess) stopIfRunning() {
	select {
	case <-p.done:
		return
	default:
		_ = p.cmd.Process.Kill()
		_ = p.waitError()
	}
}

func (p *canonicalTestProcess) waitReady(t *testing.T, httpAddress string) {
	t.Helper()
	client := &http.Client{Timeout: 300 * time.Millisecond}
	deadline := time.Now().Add(15 * time.Second)
	url := "http://" + httpAddress + "/readyz"
	for time.Now().Before(deadline) {
		select {
		case <-p.done:
			err := p.waitError()
			t.Fatalf("canonical helper exited before readiness: %v\n%s", err, p.output.String())
		default:
		}
		response, err := client.Get(url)
		if err == nil {
			_, _ = io.Copy(io.Discard, response.Body)
			_ = response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	_ = p.cmd.Process.Kill()
	_ = p.waitError()
	t.Fatalf("canonical helper did not become ready\n%s", p.output.String())
}

func (p *canonicalTestProcess) kill(t *testing.T) {
	t.Helper()
	if err := p.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := p.waitError(); err == nil {
		t.Fatal("SIGKILL helper exited successfully")
	}
}

func (p *canonicalTestProcess) terminate(t *testing.T) {
	t.Helper()
	if err := p.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	select {
	case <-p.done:
		err := p.waitError()
		if err != nil {
			t.Fatalf("canonical helper failed graceful shutdown: %v\n%s", err, p.output.String())
		}
	case <-time.After(20 * time.Second):
		_ = p.cmd.Process.Kill()
		_ = p.waitError()
		t.Fatalf("canonical helper did not shut down\n%s", p.output.String())
	}
}

func canonicalJSONRequest(t *testing.T, method, url string, payload []byte) (*http.Response, []byte) {
	t.Helper()
	request, err := http.NewRequest(method, url, bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+canonicalProcessAPIToken)
	request.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(response.Body)
	_ = response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	return response, body
}

func restartCanonicalAfterSIGKILL(
	t *testing.T,
	process *canonicalTestProcess,
	dataDir, httpAddress, grpcAddress string,
) *canonicalTestProcess {
	t.Helper()
	process.kill(t)
	restarted := startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	restarted.waitReady(t, httpAddress)
	return restarted
}

func canonicalSnapshotV2AppliedLSN(t *testing.T, dataDir string) uint64 {
	t.Helper()
	base := filepath.Join(dataDir, "index.gob.collections")
	f, err := os.Open(base + ".snapshot")
	if err != nil {
		t.Fatalf("open canonical snapshot: %v", err)
	}
	defer f.Close()

	var header [40]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		t.Fatalf("read canonical snapshot v2 header: %v", err)
	}
	if got := string(header[:8]); got != "DDCOLSNP" {
		t.Fatalf("canonical snapshot magic = %q, want %q", got, "DDCOLSNP")
	}
	if got := binary.BigEndian.Uint32(header[8:12]); got != 2 {
		t.Fatalf("canonical snapshot version = %d, want 2", got)
	}
	if got := binary.BigEndian.Uint32(header[12:16]); got != 56 {
		t.Fatalf("canonical snapshot v2 header size = %d, want 56", got)
	}
	return binary.BigEndian.Uint64(header[32:40])
}

func assertCanonicalRecoveryRetainsWAL(t *testing.T, dataDir string) {
	t.Helper()
	base := filepath.Join(dataDir, "index.gob.collections")
	_ = canonicalSnapshotV2AppliedLSN(t, dataDir)
	retained := false
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		info, err := os.Stat(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatalf("stat recovered journal %s: %v", path, err)
		}
		if !info.Mode().IsRegular() {
			t.Fatalf("recovered journal %s is not a regular file", path)
		}
		if info.Size() == 0 {
			t.Fatalf("recovered journal %s is empty", path)
		}
		retained = true
	}
	if !retained {
		t.Fatal("recovery removed all WAL evidence before an explicit checkpoint or graceful close")
	}
}

func assertCanonicalSnapshotOnlyCheckpoint(t *testing.T, dataDir string, wantAppliedLSN uint64) {
	t.Helper()
	if got := canonicalSnapshotV2AppliedLSN(t, dataDir); got != wantAppliedLSN {
		t.Fatalf("canonical snapshot applied LSN = %d, want %d", got, wantAppliedLSN)
	}
	base := filepath.Join(dataDir, "index.gob.collections")
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("snapshot-only checkpoint retained covered journal %s: %v", path, err)
		}
	}
}

func assertCanonicalHTTPMutationState(
	t *testing.T,
	baseURL string,
	wantIDs []uint64,
	temporaryExists bool,
) {
	t.Helper()
	response, body := canonicalJSONRequest(t, http.MethodGet, baseURL+"/docs", nil)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("get recovered collection returned %d: %s", response.StatusCode, body)
	}
	var collectionResult struct {
		Collection struct {
			DocCount int `json:"doc_count"`
		} `json:"collection"`
	}
	if err := json.Unmarshal(body, &collectionResult); err != nil {
		t.Fatal(err)
	}
	if collectionResult.Collection.DocCount != len(wantIDs) {
		t.Fatalf("recovered document count = %d, want %d", collectionResult.Collection.DocCount, len(wantIDs))
	}

	search := []byte(`{"queries":{"embedding":[1,0]},"top_k":10}`)
	response, body = canonicalJSONRequest(t, http.MethodPost, baseURL+"/docs/search", search)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("search recovered collection returned %d: %s", response.StatusCode, body)
	}
	var result tenantSearchJSONResponse
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatal(err)
	}
	gotIDs := make([]uint64, len(result.Documents))
	for i := range result.Documents {
		gotIDs[i] = result.Documents[i].ID
	}
	sort.Slice(gotIDs, func(i, j int) bool { return gotIDs[i] < gotIDs[j] })
	wantIDs = append([]uint64(nil), wantIDs...)
	sort.Slice(wantIDs, func(i, j int) bool { return wantIDs[i] < wantIDs[j] })
	if len(gotIDs) != len(wantIDs) {
		t.Fatalf("recovered document IDs = %v, want %v", gotIDs, wantIDs)
	}
	for i := range wantIDs {
		if gotIDs[i] != wantIDs[i] {
			t.Fatalf("recovered document IDs = %v, want %v", gotIDs, wantIDs)
		}
	}

	response, body = canonicalJSONRequest(t, http.MethodGet, baseURL+"/temporary", nil)
	if temporaryExists && response.StatusCode != http.StatusOK {
		t.Fatalf("temporary collection missing after recovery: %d: %s", response.StatusCode, body)
	}
	if !temporaryExists && response.StatusCode != http.StatusNotFound {
		t.Fatalf("deleted temporary collection reappeared after recovery: %d: %s", response.StatusCode, body)
	}
}

func TestCanonicalHTTPAckSurvivesSIGKILLAndReleasesLifetimeLock(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	httpAddress := unusedLoopbackAddress(t)
	grpcAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	defer func() { process.stopIfRunning() }()
	process.waitReady(t, httpAddress)

	baseURL := "http://" + httpAddress + "/v3/tenants/acme/collections"
	schema := []byte(`{"name":"docs","fields":[{"name":"embedding","type":"dense","dim":2,"index":{"type":"flat"}}]}`)
	response, body := canonicalJSONRequest(t, http.MethodPost, baseURL, schema)
	if response.StatusCode != http.StatusCreated {
		t.Fatalf("create returned %d: %s", response.StatusCode, body)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertCanonicalHTTPMutationState(t, baseURL, nil, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	document := []byte(`{"id":91,"vectors":{"embedding":[1,0]},"metadata":{"source":"sigkill"}}`)
	response, body = canonicalJSONRequest(t, http.MethodPost, baseURL+"/docs/docs", document)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("insert returned %d: %s", response.StatusCode, body)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertCanonicalHTTPMutationState(t, baseURL, []uint64{91}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	batch := []byte(`{"documents":[{"id":92,"vectors":{"embedding":[0,1]}},{"id":93,"vectors":{"embedding":[1,1]}}]}`)
	response, body = canonicalJSONRequest(t, http.MethodPost, baseURL+"/docs/docs/batch", batch)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("batch insert returned %d: %s", response.StatusCode, body)
	}
	var batchResult struct {
		IDs      []uint64 `json:"ids"`
		Inserted int      `json:"inserted"`
	}
	if err := json.Unmarshal(body, &batchResult); err != nil {
		t.Fatal(err)
	}
	if batchResult.Inserted != 2 || len(batchResult.IDs) != 2 || batchResult.IDs[0] != 92 || batchResult.IDs[1] != 93 {
		t.Fatalf("batch acknowledgement = %+v", batchResult)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertCanonicalHTTPMutationState(t, baseURL, []uint64{91, 92, 93}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	response, body = canonicalJSONRequest(t, http.MethodDelete, baseURL+"/docs/docs", []byte(`{"doc_id":92}`))
	if response.StatusCode != http.StatusOK {
		t.Fatalf("delete document returned %d: %s", response.StatusCode, body)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertCanonicalHTTPMutationState(t, baseURL, []uint64{91, 93}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	temporarySchema := []byte(`{"name":"temporary","fields":[{"name":"embedding","type":"dense","dim":2,"index":{"type":"flat"}}]}`)
	response, body = canonicalJSONRequest(t, http.MethodPost, baseURL, temporarySchema)
	if response.StatusCode != http.StatusCreated {
		t.Fatalf("create temporary collection returned %d: %s", response.StatusCode, body)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertCanonicalHTTPMutationState(t, baseURL, []uint64{91, 93}, true)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	response, body = canonicalJSONRequest(t, http.MethodDelete, baseURL+"/temporary", nil)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("delete collection returned %d: %s", response.StatusCode, body)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertCanonicalHTTPMutationState(t, baseURL, []uint64{91, 93}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	// Recovery deliberately retains validated WAL. A graceful close is the
	// checkpoint boundary: it must commit all six mutations to v2 and remove the
	// now-covered journals before a snapshot-only restart.
	process.terminate(t)
	assertCanonicalSnapshotOnlyCheckpoint(t, dataDir, 6)
	process = startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	process.waitReady(t, httpAddress)
	assertCanonicalHTTPMutationState(t, baseURL, []uint64{91, 93}, false)
	assertCanonicalSnapshotOnlyCheckpoint(t, dataDir, 6)
	process.terminate(t)
	assertCanonicalSnapshotOnlyCheckpoint(t, dataDir, 6)

	base := filepath.Join(dataDir, "index.gob.collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("graceful shutdown retained lifetime lock: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCanonicalGRPCAckSurvivesSIGKILL(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	httpAddress := unusedLoopbackAddress(t)
	grpcAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	defer func() { process.stopIfRunning() }()
	process.waitReady(t, httpAddress)

	connect := func() (*grpc.ClientConn, deepdatav3.DeepDataClient) {
		connection, err := grpc.NewClient(grpcAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatal(err)
		}
		return connection, deepdatav3.NewDeepDataClient(connection)
	}
	requestContext := func() (context.Context, context.CancelFunc) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		return metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+canonicalProcessAPIToken), cancel
	}
	assertState := func(wantIDs []uint64, temporaryExists bool) {
		t.Helper()
		connection, client := connect()
		defer connection.Close()

		ctx, cancel := requestContext()
		got, err := client.GetCollection(ctx, &deepdatav3.GetCollectionRequest{
			TenantId: "acme", Name: "grpc-docs",
		})
		cancel()
		if err != nil || got.GetCollection().GetDocumentCount() != uint64(len(wantIDs)) {
			t.Fatalf("gRPC recovered collection = %+v, err=%v, want count %d", got, err, len(wantIDs))
		}

		ctx, cancel = requestContext()
		searched, err := client.Search(ctx, &deepdatav3.SearchRequest{
			TenantId:   "acme",
			Collection: "grpc-docs",
			TopK:       10,
			Queries: map[string]*deepdatav3.VectorData{
				"embedding": denseProtoVector(1, 0),
			},
		})
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		gotIDs := make([]uint64, len(searched.Results))
		for i := range searched.Results {
			gotIDs[i] = searched.Results[i].Id
		}
		sort.Slice(gotIDs, func(i, j int) bool { return gotIDs[i] < gotIDs[j] })
		wantIDs = append([]uint64(nil), wantIDs...)
		sort.Slice(wantIDs, func(i, j int) bool { return wantIDs[i] < wantIDs[j] })
		if len(gotIDs) != len(wantIDs) {
			t.Fatalf("gRPC recovered document IDs = %v, want %v", gotIDs, wantIDs)
		}
		for i := range wantIDs {
			if gotIDs[i] != wantIDs[i] {
				t.Fatalf("gRPC recovered document IDs = %v, want %v", gotIDs, wantIDs)
			}
		}

		ctx, cancel = requestContext()
		listed, err := client.ListCollections(ctx, &deepdatav3.ListCollectionsRequest{TenantId: "acme"})
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		foundTemporary := false
		for _, collection := range listed.Collections {
			if collection.Name == "grpc-temporary" {
				foundTemporary = true
			}
		}
		if foundTemporary != temporaryExists {
			t.Fatalf("gRPC temporary collection present=%v, want %v; collections=%+v", foundTemporary, temporaryExists, listed.Collections)
		}
	}

	connection, client := connect()
	ctx, cancel := requestContext()
	_, err := client.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId: "acme",
		Name:     "grpc-docs",
		Fields: []*deepdatav3.VectorFieldConfig{{
			Name: "embedding", Type: int32(vcollection.VectorTypeDense), Dim: 2, IndexType: "flat",
		}},
	})
	cancel()
	if err != nil {
		connection.Close()
		t.Fatal(err)
	}
	connection.Close()
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertState(nil, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	connection, client = connect()
	ctx, cancel = requestContext()
	inserted, err := client.Insert(ctx, &deepdatav3.InsertRequest{
		TenantId:   "acme",
		Collection: "grpc-docs",
		Id:         123,
		Vectors: map[string]*deepdatav3.VectorData{
			"embedding": denseProtoVector(1, 0),
		},
	})
	cancel()
	if err != nil || inserted.Id != 123 {
		connection.Close()
		t.Fatalf("gRPC insert response=%+v err=%v", inserted, err)
	}
	connection.Close()
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertState([]uint64{123}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	connection, client = connect()
	ctx, cancel = requestContext()
	batch, err := client.BatchInsert(ctx, &deepdatav3.BatchInsertRequest{
		TenantId:   "acme",
		Collection: "grpc-docs",
		Docs: []*deepdatav3.BatchDoc{
			{Id: 124, Vectors: map[string]*deepdatav3.VectorData{"embedding": denseProtoVector(0, 1)}},
			{Id: 125, Vectors: map[string]*deepdatav3.VectorData{"embedding": denseProtoVector(1, 1)}},
		},
	})
	cancel()
	connection.Close()
	if err != nil || batch.Inserted != 2 || len(batch.Ids) != 2 || batch.Ids[0] != 124 || batch.Ids[1] != 125 {
		t.Fatalf("gRPC batch response=%+v err=%v", batch, err)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertState([]uint64{123, 124, 125}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	connection, client = connect()
	ctx, cancel = requestContext()
	_, err = client.DeleteDoc(ctx, &deepdatav3.DeleteDocRequest{
		TenantId: "acme", Collection: "grpc-docs", DocId: 124,
	})
	cancel()
	connection.Close()
	if err != nil {
		t.Fatalf("gRPC delete document: %v", err)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertState([]uint64{123, 125}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	connection, client = connect()
	ctx, cancel = requestContext()
	_, err = client.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId: "acme",
		Name:     "grpc-temporary",
		Fields: []*deepdatav3.VectorFieldConfig{{
			Name: "embedding", Type: int32(vcollection.VectorTypeDense), Dim: 2, IndexType: "flat",
		}},
	})
	cancel()
	connection.Close()
	if err != nil {
		t.Fatalf("gRPC create temporary collection: %v", err)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertState([]uint64{123, 125}, true)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	connection, client = connect()
	ctx, cancel = requestContext()
	_, err = client.DeleteCollection(ctx, &deepdatav3.DeleteCollectionRequest{
		TenantId: "acme", Name: "grpc-temporary",
	})
	cancel()
	connection.Close()
	if err != nil {
		t.Fatalf("gRPC delete collection: %v", err)
	}
	process = restartCanonicalAfterSIGKILL(t, process, dataDir, httpAddress, grpcAddress)
	assertState([]uint64{123, 125}, false)
	assertCanonicalRecoveryRetainsWAL(t, dataDir)

	process.terminate(t)
	assertCanonicalSnapshotOnlyCheckpoint(t, dataDir, 6)
	process = startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	process.waitReady(t, httpAddress)
	assertState([]uint64{123, 125}, false)
	assertCanonicalSnapshotOnlyCheckpoint(t, dataDir, 6)
	process.terminate(t)
	assertCanonicalSnapshotOnlyCheckpoint(t, dataDir, 6)
}

func TestCanonicalStartupRejectsCorruptCollectionJournalWithoutRewritingIt(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	httpAddress := unusedLoopbackAddress(t)
	grpcAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	defer func() { process.stopIfRunning() }()
	process.waitReady(t, httpAddress)

	baseURL := "http://" + httpAddress + "/v3/tenants/acme/collections"
	schema := []byte(`{"name":"docs","fields":[{"name":"embedding","type":"dense","dim":2,"index":{"type":"flat"}}]}`)
	response, body := canonicalJSONRequest(t, http.MethodPost, baseURL, schema)
	if response.StatusCode != http.StatusCreated {
		t.Fatalf("create returned %d: %s", response.StatusCode, body)
	}
	document := []byte(`{"id":71,"vectors":{"embedding":[1,0]}}`)
	response, body = canonicalJSONRequest(t, http.MethodPost, baseURL+"/docs/docs", document)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("insert returned %d: %s", response.StatusCode, body)
	}
	process.kill(t)

	base := filepath.Join(dataDir, "index.gob.collections")
	snapshotPath := base + ".snapshot"
	snapshotBefore, err := os.ReadFile(snapshotPath)
	if err != nil {
		t.Fatalf("read initialized snapshot: %v", err)
	}
	journalPath := base + ".journal"
	journal, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(journal) == 0 {
		t.Fatal("acknowledged journal is empty")
	}
	corrupt := append([]byte(nil), journal...)
	corrupt[len(corrupt)-1] ^= 0xff
	if err := os.WriteFile(journalPath, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}

	process = startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	select {
	case <-process.done:
		if err := process.waitError(); err == nil {
			t.Fatalf("corrupt-journal helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("corrupt-journal helper did not fail startup\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "checksum mismatch") {
		t.Fatalf("unexpected corrupt-journal startup error:\n%s", process.output.String())
	}
	after, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, corrupt) {
		t.Fatal("canonical startup rewrote corrupt journal evidence")
	}
	snapshotAfter, err := os.ReadFile(snapshotPath)
	if err != nil {
		t.Fatalf("read snapshot after corrupt startup: %v", err)
	}
	if !bytes.Equal(snapshotAfter, snapshotBefore) {
		t.Fatal("canonical startup rewrote the last valid snapshot after journal corruption")
	}
}

func TestCanonicalStartupBindFailureReleasesLifetimeLock(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	// Occupy the wildcard, not loopback: an authenticated process binds the
	// wildcard (canonicalListenerAddresses), and BSD/darwin honors the
	// SO_REUSEADDR that Go sets by allowing a wildcard bind alongside a
	// specific one. Only Linux calls that a conflict, so a loopback squatter
	// would let the helper start and the test would fail off Linux.
	occupied, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	defer occupied.Close()
	httpAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(t, dataDir, httpAddress, occupied.Addr().String())
	defer process.stopIfRunning()

	select {
	case <-process.done:
		err := process.waitError()
		if err == nil {
			t.Fatalf("listener-conflicted helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("listener-conflicted helper did not exit\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "complete API listener set") {
		t.Fatalf("helper failed for an unexpected reason:\n%s", process.output.String())
	}
	rebound, err := net.Listen("tcp", httpAddress)
	if err != nil {
		t.Fatalf("HTTP listener leaked after gRPC bind failure: %v", err)
	}
	rebound.Close()

	base := filepath.Join(dataDir, "index.gob.collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("listener startup failure retained lifetime lock: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCanonicalStartupRejectsSharedHTTPAndGRPCPort(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	address := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(t, dataDir, address, address)
	defer process.stopIfRunning()
	select {
	case <-process.done:
		err := process.waitError()
		if err == nil {
			t.Fatalf("shared-port helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("shared-port helper did not exit\n%s", process.output.String())
	}

	rebound, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatalf("shared listener remained bound: %v", err)
	}
	rebound.Close()
	base := filepath.Join(dataDir, "index.gob.collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("shared-port refusal retained lifetime lock: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCanonicalStartupRefusesLegacyV2StateWithoutRewritingIt(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(dataDir, "index.gob.collections")
	manager := vcollection.NewCollectionManager(base)
	if _, err := manager.CreateCollection(context.Background(), vcollection.CollectionSchema{
		Name: "legacy",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	metadata, err := vcollection.NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := vcollection.SaveUnifiedCollectionSnapshot(
		base, manager, vcollection.NewTenantManager(base), metadata,
	); err != nil {
		t.Fatal(err)
	}
	wantSnapshot, err := os.ReadFile(base + ".snapshot")
	if err != nil {
		t.Fatal(err)
	}

	process := startCanonicalTestProcess(t, dataDir, "127.0.0.1:1", "127.0.0.1:2")
	defer process.stopIfRunning()
	select {
	case <-process.done:
		if err := process.waitError(); err == nil {
			t.Fatalf("legacy V2 helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("legacy V2 helper did not exit\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "legacy V2 collections") {
		t.Fatalf("helper failed for an unexpected reason:\n%s", process.output.String())
	}
	gotSnapshot, err := os.ReadFile(base + ".snapshot")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotSnapshot, wantSnapshot) {
		t.Fatal("canonical startup refusal rewrote legacy V2 snapshot")
	}

	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("legacy refusal retained lifetime lock: %v", err)
	}
	got, err := store.LegacyCollectionCount()
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("legacy snapshot collection count = %d, want 1", got)
	}
	if err := store.Abort(); err != nil {
		t.Fatal(err)
	}
}

func TestCanonicalStartupRefusesRawLegacyV2StateWithoutMutation(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(dataDir, "index.gob.collections")
	manager := vcollection.NewCollectionManager(base)
	if _, err := manager.CreateCollection(context.Background(), vcollection.CollectionSchema{
		Name: "legacy",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := manager.Save(base + ".manager"); err != nil {
		t.Fatal(err)
	}
	tenants := vcollection.NewTenantManager(base)
	if _, err := tenants.CreateCollection(context.Background(), "tenant", vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tenants.Save(base + ".tenants"); err != nil {
		t.Fatal(err)
	}

	rawPaths := []string{base + ".manager", base + ".tenants"}
	before := make(map[string][32]byte, len(rawPaths))
	for _, path := range rawPaths {
		before[path] = testFileSHA256(t, path)
	}

	// Raw legacy preflight happens before either API listener is bound, so fixed
	// unused test ports are sufficient and avoid a reserve/close race here.
	process := startCanonicalTestProcess(t, dataDir, "127.0.0.1:1", "127.0.0.1:2")
	defer process.stopIfRunning()
	select {
	case <-process.done:
		if err := process.waitError(); err == nil {
			t.Fatalf("raw legacy V2 helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("raw legacy V2 helper did not exit\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "legacy V2 collection persistence") {
		t.Fatalf("helper failed for an unexpected reason:\n%s", process.output.String())
	}
	for _, path := range rawPaths {
		if got := testFileSHA256(t, path); got != before[path] {
			t.Fatalf("canonical startup changed raw legacy artifact %s", path)
		}
	}
	for _, path := range []string{
		base + ".snapshot",
		base + ".initialized",
		base + ".lock",
		base + ".journal",
		base + ".journal.frozen",
	} {
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("canonical raw legacy refusal created %s: %v", path, err)
		}
	}
}

func TestCanonicalStartupRefusesLegacyRootStateWithoutMutation(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// The root guard refuses on artifact existence alone; content is opaque
	// legacy gob/WAL data that must never be read, rewritten, or removed.
	rawPaths := []string{
		filepath.Join(dataDir, "index.gob"),
		filepath.Join(dataDir, "index.gob.wal"),
		filepath.Join(dataDir, "index.gob.wal.frozen"),
	}
	for i, path := range rawPaths {
		if err := os.WriteFile(path, []byte(fmt.Sprintf("legacy-root-artifact-%d", i)), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	before := make(map[string][32]byte, len(rawPaths))
	for _, path := range rawPaths {
		before[path] = testFileSHA256(t, path)
	}

	// Root preflight happens before either API listener is bound, so fixed
	// unused test ports are sufficient here as well.
	process := startCanonicalTestProcess(t, dataDir, "127.0.0.1:1", "127.0.0.1:2")
	defer process.stopIfRunning()
	select {
	case <-process.done:
		if err := process.waitError(); err == nil {
			t.Fatalf("legacy root helper exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("legacy root helper did not exit\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "legacy root persistence") {
		t.Fatalf("helper failed for an unexpected reason:\n%s", process.output.String())
	}
	for _, path := range rawPaths {
		if got := testFileSHA256(t, path); got != before[path] {
			t.Fatalf("canonical startup changed legacy root artifact %s", path)
		}
	}
	entries, err := os.ReadDir(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != len(rawPaths) {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("canonical legacy root refusal created new artifacts: %v", names)
	}
}
