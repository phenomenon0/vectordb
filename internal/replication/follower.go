package replication

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// ErrResyncRequired means the leader no longer retains the records this
// replica needs: a checkpoint covered and removed them while the replica was
// behind or stopped.
//
// Recovering means discarding the local replica directory and bootstrapping
// again. This package will not do that on its own. A replica directory is real
// state -- it may be the one a read fleet is serving from, and it may be the
// only copy of a leader generation someone is about to inspect -- so deleting
// it is an operator's decision, made once, not a client's retry policy
// executing silently at 3am.
var ErrResyncRequired = errors.New("replica is behind the leader's retained journal and must be re-bootstrapped")

// ErrUnknownTenant means a tenant-scoped request got a 404 from the leader:
// the leader no longer serves this tenant, most likely because it was
// deleted. Terminal like ErrResyncRequired -- reconnecting will not help.
var ErrUnknownTenant = errors.New("leader does not know this tenant")

// Follower pulls a leader's journal into a local read replica.
type Follower struct {
	// LeaderURL is the leader's base URL, e.g. https://leader.internal:8080.
	LeaderURL string
	// Token is the node credential. It must match the leader's.
	Token string
	// Tenant selects one tenant on a per-tenant leader (NewTenantLeaderHandler):
	// Status, Open and Follow route to PathPrefix+"tenants/"+Tenant+"/"+<call>
	// instead of the bare route, which a per-tenant leader answers with 404;
	// Tenants does not need it.
	Tenant string
	// Client defaults to http.DefaultClient. A follow stream is open-ended, so
	// a custom client must not set a whole-request Timeout.
	Client *http.Client
	// OnPreamble, if set, is called with the leader's stream preamble right
	// after ReadPreamble succeeds and the StoreID matched. A standby uses it
	// to learn the leader's LatestLSN for its lag report; nil is ignored.
	OnPreamble func(Preamble)
}

// LeaderStatus is what the leader reports about itself.
type LeaderStatus struct {
	ProtocolVersion int    `json:"protocol_version"`
	StoreID         string `json:"store_id"`
	LatestLSN       uint64 `json:"latest_lsn"`
}

// ID decodes the leader's store ID.
func (s LeaderStatus) ID() ([16]byte, error) {
	var id [16]byte
	raw, err := hex.DecodeString(s.StoreID)
	if err != nil || len(raw) != 16 {
		return id, fmt.Errorf("leader reported an unusable store ID %q", s.StoreID)
	}
	copy(id[:], raw)
	return id, nil
}

func (f *Follower) client() *http.Client {
	if f.Client != nil {
		return f.Client
	}
	return http.DefaultClient
}

// get issues an authenticated GET for one of the three per-store calls
// (status, snapshot, journal). With Tenant set it targets that tenant's
// route on a per-tenant leader; empty targets the bare route a single-store
// leader (NewLeaderHandler) still serves.
func (f *Follower) get(ctx context.Context, path string, query url.Values) (*http.Response, error) {
	if f.Tenant != "" {
		path = "tenants/" + f.Tenant + "/" + path
	}
	return f.doGet(ctx, path, query)
}

// Tenants lists every tenant ID the leader currently replicates, sorted.
func (f *Follower) Tenants(ctx context.Context) ([]string, error) {
	resp, err := f.doGet(ctx, "tenants", nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	var body struct {
		Tenants []string `json:"tenants"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode leader tenants: %w", err)
	}
	return body.Tenants, nil
}

func (f *Follower) doGet(ctx context.Context, path string, query url.Values) (*http.Response, error) {
	base := strings.TrimSuffix(f.LeaderURL, "/")
	target := base + PathPrefix + path
	if len(query) > 0 {
		target += "?" + query.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+f.Token)
	resp, err := f.client().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		_ = resp.Body.Close()
		detail := fmt.Errorf("leader %s returned %s: %s", path, resp.Status, strings.TrimSpace(string(body)))
		// 409 is the leader's one pre-stream rejection: the cursor names a
		// store it is not. That is a misconfiguration -- this follower is
		// pointed at the wrong leader -- and must be distinguishable from the
		// transport failures a caller should retry through.
		if resp.StatusCode == http.StatusConflict {
			return nil, fmt.Errorf("%w: %s", vcollection.ErrJournalStoreMismatch, detail)
		}
		// A tenant-scoped route only ever answers 404 for "unknown tenant"
		// (route() in leader.go) -- distinguish it from a transport failure a
		// caller should retry through.
		if resp.StatusCode == http.StatusNotFound && f.Tenant != "" {
			return nil, fmt.Errorf("%w: %s", ErrUnknownTenant, detail)
		}
		return nil, detail
	}
	return resp, nil
}

// Status asks the leader who it is and how far its journal has been written.
func (f *Follower) Status(ctx context.Context) (LeaderStatus, error) {
	resp, err := f.get(ctx, "status", nil)
	if err != nil {
		return LeaderStatus{}, err
	}
	defer func() { _ = resp.Body.Close() }()
	var status LeaderStatus
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<10)).Decode(&status); err != nil {
		return LeaderStatus{}, fmt.Errorf("decode leader status: %w", err)
	}
	if status.ProtocolVersion != Version {
		return status, fmt.Errorf("%w: leader speaks version %d, this build speaks %d", ErrVersionMismatch, status.ProtocolVersion, Version)
	}
	return status, nil
}

// Open returns a replica store at basePath, bootstrapping it from the leader's
// snapshot when the path holds no store yet and resuming an existing one
// otherwise.
//
// It is Seed + vcollection.OpenDurableStore + Bind: Seed does the bootstrap
// (or nothing, if the path is already occupied), the plain open resumes
// whatever is on disk, and Bind applies the same StoreID guard and marker a
// bootstrap already carries. A caller that owns a StoreSet-opened store
// instead of a bare path uses Seed and Bind directly (see StoreSet.AdoptTenant).
func (f *Follower) Open(ctx context.Context, basePath, storagePath string) (*vcollection.DurableStore, error) {
	if _, err := f.Seed(ctx, basePath, storagePath); err != nil {
		return nil, err
	}
	store, err := vcollection.OpenDurableStore(basePath, storagePath)
	if err != nil {
		return nil, fmt.Errorf("open existing replica: %w", err)
	}
	if err := f.Bind(ctx, store, basePath); err != nil {
		// Abort, not Close: Close would checkpoint a store we are rejecting.
		return nil, errors.Join(err, store.Abort())
	}
	return store, nil
}

// Seed bootstraps a replica store at basePath from the leader's snapshot when
// the path holds no store yet, and does nothing when it is already occupied
// -- the emptiness test is the same prefix scan BootstrapReplica uses, so the
// two cannot disagree about what an empty target is and accidentally
// bootstrap over a live replica.
//
// The store is marked a replica and released (Abort, not Close: a checkpoint
// here would be pointless work on a store nobody has opened for real yet)
// before Seed returns, so the directory is safe for a later plain open. The
// marker has to land before that release: an unmarked seeded directory is
// exactly the directory marker.go's package comment exists to prevent -- one
// that looks ordinary on disk and would accept local writes if something
// opened it first.
func (f *Follower) Seed(ctx context.Context, basePath, storagePath string) (bootstrapped bool, err error) {
	occupied, err := storeArtifactsExist(basePath)
	if err != nil {
		return false, err
	}
	if occupied {
		return false, nil
	}
	status, err := f.Status(ctx)
	if err != nil {
		return false, err
	}
	leaderID, err := status.ID()
	if err != nil {
		return false, err
	}
	store, err := f.bootstrap(ctx, basePath, storagePath, leaderID)
	if err != nil {
		return false, err
	}
	if err := markOrAbort(store, basePath, leaderID); err != nil {
		return false, err
	}
	return true, store.Abort()
}

// Bind asserts that store already holds the leader's StoreID and marks it a
// replica on disk. It is the guard an occupied Open resumes through, split
// out so a StoreSet-owned store -- one this package never opened and must not
// abort on failure -- can be bound the same way before it starts serving.
//
// The caller owns store: unlike Open and Seed, Bind never calls Abort. A
// mismatch or a failed MakeReplica leaves the store exactly as it was handed
// in, for the caller to close or abort as it sees fit.
func (f *Follower) Bind(ctx context.Context, store *vcollection.DurableStore, basePath string) error {
	status, err := f.Status(ctx)
	if err != nil {
		return err
	}
	leaderID, err := status.ID()
	if err != nil {
		return err
	}
	// The same trust boundary BootstrapReplica applies after its own open. A
	// replica adopts the leader's StoreID out of the snapshot header, so a
	// store sitting here under a different identity is somebody else's data --
	// an unrelated single-node store, or a replica of a different leader. Only
	// MakeReplica's own guard would catch the second case, and neither guard
	// catches the first: it would promote that store and start appending this
	// leader's records on top of a history they were never part of.
	if id := store.Metadata().StoreID; id != leaderID {
		return fmt.Errorf("%w: %q holds store %x, leader is %x", vcollection.ErrJournalStoreMismatch, basePath, id, leaderID)
	}
	// MakeReplica before anything else can write: an unmarked store would
	// accept a local write, consume the LSN the leader's next record needs, and
	// wedge this replica permanently.
	if err := store.MakeReplica(leaderID); err != nil {
		return fmt.Errorf("bind replica to leader %x: %w", leaderID, err)
	}
	return MarkReplica(basePath, leaderID)
}

// markOrAbort persists the replica binding, or gives the directory up.
//
// Both of Open's paths end here, because both produce the same thing: a
// directory that is a replica and whose next reader may be a `deepdata serve`
// in another process. Returning a store whose marker did not land would leave
// exactly the directory this marker exists to prevent -- one that looks
// ordinary and takes writes.
//
// Abort rather than Close on failure: Close would checkpoint state we are
// refusing to vouch for. The store artifacts stay behind, so the next run
// takes Open's resume path and retries the marker.
func markOrAbort(store *vcollection.DurableStore, basePath string, leaderID [16]byte) error {
	if err := MarkReplica(basePath, leaderID); err != nil {
		return errors.Join(err, store.Abort())
	}
	return nil
}

func (f *Follower) bootstrap(ctx context.Context, basePath, storagePath string, leaderID [16]byte) (*vcollection.DurableStore, error) {
	resp, err := f.get(ctx, "snapshot", nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	// A cheap reject before writing the whole transfer to disk. It is not the
	// trust boundary -- BootstrapReplica re-checks the ID inside the snapshot's
	// own checksummed header, which a header cannot forge.
	if advertised := resp.Header.Get("X-Deepdata-Store-Id"); advertised != "" {
		if advertised != hex.EncodeToString(leaderID[:]) {
			return nil, fmt.Errorf("%w: snapshot advertises store %s, status said %x", vcollection.ErrJournalStoreMismatch, advertised, leaderID)
		}
	}
	store, err := vcollection.BootstrapReplica(basePath, storagePath, resp.Body, leaderID)
	if err != nil {
		return nil, fmt.Errorf("bootstrap replica from leader snapshot: %w", err)
	}
	return store, nil
}

// Follow applies the leader's records to store until ctx is canceled or the
// stream cannot continue.
//
// It resumes from the store's own durable cursor, so a restarted replica picks
// up exactly where it stopped. A returned ErrResyncRequired means the leader
// has discarded the records this replica still needs; every other error is the
// stream failing and is safe to retry from the same cursor.
func (f *Follower) Follow(ctx context.Context, store *vcollection.DurableStore) error {
	if store == nil {
		return errors.New("replication follower needs a replica store")
	}
	if !store.IsReplica() {
		return vcollection.ErrReplicaNotConfigured
	}
	cursor := store.ReplicaCursor()
	query := url.Values{
		"after":  {fmt.Sprint(cursor.LSN)},
		"follow": {"1"},
	}
	// Sent only once the replica has an identity to assert. On a first sync
	// after bootstrap the cursor already carries the leader's ID, so this is
	// always set in practice; the guard keeps a zero cursor from asking the
	// leader to match all-zero bytes.
	if cursor.StoreID != ([16]byte{}) {
		query.Set("store", hex.EncodeToString(cursor.StoreID[:]))
	}
	resp, err := f.get(ctx, "journal", query)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	pre, err := ReadPreamble(resp.Body)
	if err != nil {
		return err
	}
	if pre.StoreID != cursor.StoreID {
		return fmt.Errorf("%w: stream is from store %x, replica follows %x", vcollection.ErrJournalStoreMismatch, pre.StoreID, cursor.StoreID)
	}
	if f.OnPreamble != nil {
		f.OnPreamble(pre)
	}

	var buf []byte
	for {
		lsn, payload, err := ReadRecord(resp.Body, &buf)
		if err != nil {
			return f.streamEnded(ctx, err)
		}
		// ApplyReplicated is the authority on ordering: it compares against the
		// journal's own next LSN and rejects a re-delivered or skipped record.
		// Repeating that comparison here would be a second opinion that can
		// drift from the one that actually guards the journal.
		if err := store.ApplyReplicated(ctx, vcollection.JournalRecord{LSN: lsn, Payload: payload}); err != nil {
			return fmt.Errorf("apply replicated record at LSN %d: %w", lsn, err)
		}
	}
}

// streamEnded classifies why the record loop stopped.
func (f *Follower) streamEnded(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	if errors.Is(err, io.EOF) {
		// A follow stream has no clean end; the leader closing one is a
		// restart or a proxy timeout, and the caller retries from the same
		// durable cursor.
		return io.ErrUnexpectedEOF
	}
	var ctl *ControlError
	if errors.As(err, &ctl) {
		if ctl.Code == ControlGap {
			return fmt.Errorf("%w: %s", ErrResyncRequired, ctl.Message)
		}
		return ctl
	}
	return err
}

// storeArtifactsExist reports whether basePath already holds a durable store.
//
// It is BootstrapReplica's emptiness predicate, spelled the same way for the
// same reason: every artifact is basePath+"."+suffix, and the dot keeps a
// sibling store whose name merely extends this one ("shard10" beside "shard1")
// from being mistaken for this one's.
func storeArtifactsExist(basePath string) (bool, error) {
	if basePath == "" {
		return false, errors.New("replica base path cannot be empty")
	}
	basePath = filepath.Clean(basePath)
	entries, err := os.ReadDir(filepath.Dir(basePath))
	if err != nil {
		return false, fmt.Errorf("inspect replica directory: %w", err)
	}
	prefix := filepath.Base(basePath) + "."
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), prefix) {
			return true, nil
		}
	}
	return false, nil
}
