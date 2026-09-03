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

// Follower pulls a leader's journal into a local read replica.
type Follower struct {
	// LeaderURL is the leader's base URL, e.g. https://leader.internal:8080.
	LeaderURL string
	// Token is the node credential. It must match the leader's.
	Token string
	// Client defaults to http.DefaultClient. A follow stream is open-ended, so
	// a custom client must not set a whole-request Timeout.
	Client *http.Client
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

func (f *Follower) get(ctx context.Context, path string, query url.Values) (*http.Response, error) {
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
// The emptiness test is the same prefix scan BootstrapReplica uses, so the two
// paths cannot disagree about what an empty target is and accidentally
// bootstrap over a live replica.
func (f *Follower) Open(ctx context.Context, basePath, storagePath string) (*vcollection.DurableStore, error) {
	status, err := f.Status(ctx)
	if err != nil {
		return nil, err
	}
	leaderID, err := status.ID()
	if err != nil {
		return nil, err
	}
	occupied, err := storeArtifactsExist(basePath)
	if err != nil {
		return nil, err
	}
	if !occupied {
		return f.bootstrap(ctx, basePath, storagePath, leaderID)
	}
	store, err := vcollection.OpenDurableStore(basePath, storagePath)
	if err != nil {
		return nil, fmt.Errorf("open existing replica: %w", err)
	}
	// The same trust boundary BootstrapReplica applies after its own open. A
	// replica adopts the leader's StoreID out of the snapshot header, so a
	// store sitting here under a different identity is somebody else's data --
	// an unrelated single-node store, or a replica of a different leader. Only
	// MakeReplica's own guard would catch the second case, and neither guard
	// catches the first: it would promote that store and start appending this
	// leader's records on top of a history they were never part of.
	if id := store.Metadata().StoreID; id != leaderID {
		// Abort, not Close: Close would checkpoint a store we are rejecting.
		return nil, errors.Join(
			fmt.Errorf("%w: %q holds store %x, leader is %x", vcollection.ErrJournalStoreMismatch, basePath, id, leaderID),
			store.Abort(),
		)
	}
	// MakeReplica before anything else can write: an unmarked store would
	// accept a local write, consume the LSN the leader's next record needs, and
	// wedge this replica permanently.
	if err := store.MakeReplica(leaderID); err != nil {
		return nil, errors.Join(fmt.Errorf("bind replica to leader %x: %w", leaderID, err), store.Abort())
	}
	return store, nil
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
