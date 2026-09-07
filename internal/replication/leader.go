package replication

import (
	"context"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// PathPrefix is the node surface. It is deliberately outside /v3: the client
// contract in api/contract/v3/operations.json describes tenant-scoped calls an
// application makes, and every one of those is authorized against a tenant.
// These are whole-store calls between servers, authorized against a separate
// node credential, and listing them beside the client operations would put a
// route that exports every tenant's data behind a tenant's own token.
const PathPrefix = "/replication/v1/"

// Source is the leader half of the store, narrowed to what the transport
// needs. Nothing here decides durability; it reads what the store already
// committed.
type Source interface {
	JournalStatus() (vcollection.JournalPosition, error)
	StreamJournal(cursor vcollection.JournalCursor, visit func(vcollection.JournalRecord) error) (vcollection.JournalPosition, error)
	FollowJournal(ctx context.Context, cursor vcollection.JournalCursor, visit func(vcollection.JournalRecord) error) error
	WriteSnapshot(w io.Writer) (vcollection.JournalPosition, error)
}

// LeaderConfig configures the node surface. A zero Token means the surface is
// off; NewLeaderHandler refuses to build a handler without one rather than
// serving whole-store reads to whoever can reach the port.
type LeaderConfig struct {
	// Token authenticates a peer node. It is NOT the client API token: the
	// snapshot route exports every tenant in one request, so it must not be
	// reachable with a tenant credential, nor through the anonymous
	// server-admin context that credentialless dev mode grants.
	Token string
	// SpoolDir is where a snapshot is staged before it is streamed. Empty uses
	// the OS temp dir.
	SpoolDir string
	// Logger receives stream-level failures. Nil discards them.
	Logger *log.Logger
}

// NewTenantLeaderHandler builds the node surface over one store per tenant.
// tenants lists the currently known tenant IDs; source resolves one of them
// to its Source, or reports it unknown. Both are called per request, so a
// caller backed by a live collection.StoreSet sees tenants as they are
// created and deleted without rebuilding this handler.
func NewTenantLeaderHandler(cfg LeaderConfig, tenants func() []string, source func(tenantID string) (Source, bool)) (http.Handler, error) {
	if tenants == nil || source == nil {
		return nil, errors.New("replication leader needs tenant sources")
	}
	if err := validateLeaderConfig(cfg); err != nil {
		return nil, err
	}
	tl := &tenantLeader{cfg: cfg, tenants: tenants, source: source}
	mux := http.NewServeMux()
	mux.HandleFunc(PathPrefix+"tenants", authed(cfg, tl.listTenants))
	mux.HandleFunc(PathPrefix+"tenants/{tenant}/status", authed(cfg, tl.route((*leader).status)))
	mux.HandleFunc(PathPrefix+"tenants/{tenant}/snapshot", authed(cfg, tl.route((*leader).snapshot)))
	mux.HandleFunc(PathPrefix+"tenants/{tenant}/journal", authed(cfg, tl.route((*leader).journal)))
	perTenantOnly := authed(cfg, tl.perTenantOnly)
	mux.HandleFunc(PathPrefix+"status", perTenantOnly)
	mux.HandleFunc(PathPrefix+"snapshot", perTenantOnly)
	mux.HandleFunc(PathPrefix+"journal", perTenantOnly)
	return mux, nil
}

// validateLeaderConfig checks the fields both leader constructors need before
// building a handler: a node token (never optional, since the snapshot route
// exports whole-tenant state) and, if given, a real spool directory.
func validateLeaderConfig(cfg LeaderConfig) error {
	if cfg.Token == "" {
		return errors.New("replication leader needs a node token; the snapshot route exports every tenant")
	}
	if cfg.SpoolDir != "" {
		info, err := os.Stat(cfg.SpoolDir)
		if err != nil {
			return fmt.Errorf("replication spool directory: %w", err)
		}
		if !info.IsDir() {
			return fmt.Errorf("replication spool directory %q is not a directory", cfg.SpoolDir)
		}
	}
	return nil
}

// tenantLeader is the per-tenant node surface. It resolves a Source per
// request rather than owning one, and dispatches to the same status/
// snapshot/journal bodies leader uses via a throwaway *leader built over the
// resolved Source, so the wire format is byte-identical to the single-store
// surface.
type tenantLeader struct {
	cfg     LeaderConfig
	tenants func() []string
	source  func(tenantID string) (Source, bool)
}

// route resolves {tenant} from the request path and dispatches to fn, or
// answers 404 when the ID is unsafe or unknown to source().
func (tl *tenantLeader) route(fn func(*leader, http.ResponseWriter, *http.Request)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("tenant")
		if !vcollection.ValidTenantID(id) {
			writeReplicationError(w, http.StatusNotFound, "unknown tenant")
			return
		}
		src, ok := tl.source(id)
		if !ok {
			writeReplicationError(w, http.StatusNotFound, "unknown tenant")
			return
		}
		fn(&leader{src: src, cfg: tl.cfg}, w, r)
	}
}

// listTenants answers every tenant ID this leader currently replicates. The
// slice is copied before sorting: tenants() may hand back a reference to
// live state (e.g. a StoreSet's internal map order), and this must not
// mutate it.
func (tl *tenantLeader) listTenants(w http.ResponseWriter, r *http.Request) {
	ids := append([]string(nil), tl.tenants()...)
	sort.Strings(ids)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"tenants": ids})
}

// perTenantOnly answers the bare, pre-multitenancy routes: this leader has no
// single store to serve them from.
func (tl *tenantLeader) perTenantOnly(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error": "this leader replicates per tenant",
		"hint":  "GET " + PathPrefix + "tenants, then " + PathPrefix + "tenants/{tenant}/status",
	})
}

// writeReplicationError writes a {"error": message} JSON body with status.
func writeReplicationError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

type leader struct {
	src Source
	cfg LeaderConfig
}

func (l *leader) logf(format string, args ...any) {
	if l.cfg.Logger != nil {
		l.cfg.Logger.Printf(format, args...)
	}
}

// authed gates every node route on cfg's node token in constant time. It
// reports nothing about why a request failed: an unauthenticated caller
// learns only that the route exists, which it can already tell from the
// port. Shared by both leader constructors so a per-tenant route and a
// single-store route reject exactly the same way.
func authed(cfg LeaderConfig, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if subtle.ConstantTimeCompare([]byte(token), []byte(cfg.Token)) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

// status lets a follower decide between bootstrap and resume without
// downloading anything.
func (l *leader) status(w http.ResponseWriter, r *http.Request) {
	pos, err := l.src.JournalStatus()
	if err != nil {
		http.Error(w, "leader journal unavailable", http.StatusServiceUnavailable)
		l.logf("replication: status: %v", err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"protocol_version": Version,
		"store_id":         hex.EncodeToString(pos.StoreID[:]),
		"latest_lsn":       pos.LatestLSN,
	})
}

// snapshot streams one complete generation for replica bootstrap.
//
// WriteSnapshot holds the store's write barrier for the whole write and its
// contract says the sink must be local, never a socket. So the generation is
// staged to a spool file under the barrier and shipped afterwards: a follower
// on a slow link stalls its own transfer, not the leader's writes.
func (l *leader) snapshot(w http.ResponseWriter, r *http.Request) {
	spool, err := os.CreateTemp(l.cfg.SpoolDir, "deepdata-replication-*.snapshot")
	if err != nil {
		http.Error(w, "snapshot spool unavailable", http.StatusServiceUnavailable)
		l.logf("replication: spool: %v", err)
		return
	}
	// Unlinked immediately: the bytes stay reachable through the open handle,
	// and a crash mid-transfer cannot leave a stray copy of every tenant's
	// state in the temp directory.
	defer func() { _ = spool.Close() }()
	if err := os.Remove(spool.Name()); err != nil {
		l.logf("replication: unlink spool %q: %v", spool.Name(), err)
	}

	pos, err := l.src.WriteSnapshot(spool)
	if err != nil {
		http.Error(w, "snapshot unavailable", http.StatusServiceUnavailable)
		l.logf("replication: snapshot: %v", err)
		return
	}
	size, err := spool.Seek(0, io.SeekEnd)
	if err == nil {
		_, err = spool.Seek(0, io.SeekStart)
	}
	if err != nil {
		http.Error(w, "snapshot spool unreadable", http.StatusInternalServerError)
		l.logf("replication: spool rewind: %v", err)
		return
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))

	// The follower checks both against what the snapshot header says, so these
	// are a fast reject, not the trust boundary. BootstrapReplica is.
	w.Header().Set("X-Deepdata-Store-Id", hex.EncodeToString(pos.StoreID[:]))
	w.Header().Set("X-Deepdata-Applied-Lsn", strconv.FormatUint(pos.LatestLSN, 10))

	// A whole-store transfer is bounded but not fast; same deadline problem as
	// a tail, minus the "forever" part.
	clearWriteDeadline(w)
	if _, err := io.Copy(w, spool); err != nil {
		// Headers are already sent; Content-Length makes the short body visible
		// to the follower, which fails its own open rather than trusting it.
		l.logf("replication: snapshot transfer: %v", err)
	}
}

// journal streams records after ?after=, optionally tailing with ?follow=1.
func (l *leader) journal(w http.ResponseWriter, r *http.Request) {
	cursor, err := parseCursor(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	pos, err := l.src.JournalStatus()
	if err != nil {
		http.Error(w, "leader journal unavailable", http.StatusServiceUnavailable)
		l.logf("replication: journal status: %v", err)
		return
	}
	// Everything that can be judged before the 200 is judged before the 200, so
	// a misconfigured follower gets an HTTP status instead of a control frame.
	if cursor.StoreID != ([16]byte{}) && cursor.StoreID != pos.StoreID {
		http.Error(w, fmt.Sprintf("cursor belongs to store %x; this leader is %x", cursor.StoreID, pos.StoreID), http.StatusConflict)
		return
	}

	// A tail has no length and no natural end, so the server's per-connection
	// write deadline -- sized for a request/response API -- would cut it. The
	// follower would read a truncated stream and resume, forever, at whatever
	// interval that deadline is.
	clearWriteDeadline(w)

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if err := WritePreamble(w, Preamble{Version: Version, StoreID: pos.StoreID, LatestLSN: pos.LatestLSN}); err != nil {
		l.logf("replication: preamble: %v", err)
		return
	}
	flusher, _ := w.(http.Flusher)
	if flusher != nil {
		// A follow stream can be idle for a long time; without this the
		// preamble sits in the buffer and the follower cannot even confirm it
		// reached the right leader.
		flusher.Flush()
	}

	send := func(record vcollection.JournalRecord) error {
		if err := WriteRecord(w, record.LSN, record.Payload); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	}

	if r.URL.Query().Get("follow") == "1" {
		err = l.src.FollowJournal(r.Context(), cursor, send)
	} else {
		_, err = l.src.StreamJournal(cursor, send)
	}
	switch {
	case err == nil, errors.Is(err, contextCanceled(r)):
		return
	case errors.Is(err, vcollection.ErrJournalGap):
		l.control(w, flusher, ControlGap, err)
	case errors.Is(err, vcollection.ErrJournalStoreMismatch):
		l.control(w, flusher, ControlStoreMismatch, err)
	default:
		l.control(w, flusher, ControlFault, err)
	}
}

// control reports a mid-stream end the follower can branch on. The status line
// is long gone by now, which is exactly why the protocol has these frames.
func (l *leader) control(w http.ResponseWriter, flusher http.Flusher, code byte, cause error) {
	l.logf("replication: stream ended (code %d): %v", code, cause)
	if err := WriteControl(w, code, cause.Error()); err != nil {
		l.logf("replication: control frame: %v", err)
		return
	}
	if flusher != nil {
		flusher.Flush()
	}
}

// contextCanceled is the request's own cancellation, which is a follower
// hanging up rather than a leader failure and must not be reported as one.
func contextCanceled(r *http.Request) error { return r.Context().Err() }

func parseCursor(r *http.Request) (vcollection.JournalCursor, error) {
	q := r.URL.Query()
	var cursor vcollection.JournalCursor
	if raw := q.Get("after"); raw != "" {
		lsn, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			return cursor, fmt.Errorf("after must be a journal LSN: %v", err)
		}
		cursor.LSN = lsn
	}
	// The store is optional on a first sync and mandatory after it: a follower
	// that has ever applied a record sends the ID it applied under, so being
	// repointed at a different leader is a 409 instead of an interleave.
	if raw := q.Get("store"); raw != "" {
		id, err := hex.DecodeString(raw)
		if err != nil || len(id) != 16 {
			return cursor, errors.New("store must be a 32-character hex store ID")
		}
		copy(cursor.StoreID[:], id)
	}
	return cursor, nil
}

// clearWriteDeadline removes the per-connection write deadline for one
// response. The error is deliberately ignored: a ResponseWriter that cannot do
// this has no deadline to clear, which is exactly the state we want.
func clearWriteDeadline(w http.ResponseWriter) {
	_ = http.NewResponseController(w).SetWriteDeadline(time.Time{})
}
