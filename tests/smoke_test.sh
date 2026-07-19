#!/usr/bin/env bash
# Canonical DeepData RC smoke: authenticated HTTP V3, all nine unary gRPC
# methods, unsupported-surface checks, graceful restart, and durable state.
set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=$(mktemp -d /tmp/deepdata-rc-smoke-XXXXXX)
STATE_ROOT="$WORK_DIR/state"
SERVER_LOG="$WORK_DIR/server.log"
HTTP_PORT=${DEEPDATA_PORT:-9777}
GRPC_PORT_NUMBER=${DEEPDATA_GRPC_PORT:-59777}
BASE_URL="http://127.0.0.1:$HTTP_PORT"
GRPC_ADDRESS="127.0.0.1:$GRPC_PORT_NUMBER"
API_TOKEN=${DEEPDATA_API_TOKEN:-canonical-smoke-test-token-strong-credential}
TENANT=smoke
SERVER_PID=""

if [[ -n "${DEEPDATA_BINARY:-}" ]]; then
  BINARY=$(realpath -e -- "$DEEPDATA_BINARY")
  [[ -x "$BINARY" ]]
else
  BINARY="$WORK_DIR/deepdata"
fi

cleanup() {
  local rc=$?
  set +e
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 100); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null
    fi
    wait "$SERVER_PID" 2>/dev/null
  fi
  if ((rc == 0)); then
    rm -rf -- "$WORK_DIR"
  else
    echo "smoke failed; retained evidence at $WORK_DIR" >&2
    tail -n 120 "$SERVER_LOG" >&2 2>/dev/null
  fi
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  return 1
}

require_command() {
  command -v "$1" >/dev/null || fail "required command not found: $1"
}

require_command curl
require_command jq
require_command go

mkdir -m 0700 -- "$STATE_ROOT"
cd -- "$ROOT_DIR"

if [[ -z "${DEEPDATA_BINARY:-}" ]]; then
  echo "building canonical server"
  go build -trimpath -o "$BINARY" ./cmd/deepdata
fi

GRPC_SOURCE="$WORK_DIR/grpc_smoke.go"
GRPC_PROBE="$WORK_DIR/grpc_smoke"
cat >"$GRPC_SOURCE" <<'GOEOF'
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func dense(values ...float32) *deepdatav3.VectorData {
	return &deepdatav3.VectorData{Data: &deepdatav3.VectorData_Dense{
		Dense: &deepdatav3.DenseVector{Values: values},
	}}
}

func containsID(results []*deepdatav3.SearchHit, id uint64) bool {
	for _, result := range results {
		if result.GetId() == id {
			return true
		}
	}
	return false
}

func main() {
	if len(os.Args) != 4 {
		panic("usage: grpc_smoke exercise|verify address token")
	}
	mode, address, token := os.Args[1], os.Args[2], os.Args[3]
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(
		ctx,
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	must(err)
	defer conn.Close()
	client := deepdatav3.NewDeepDataClient(conn)
	ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+token)

	const tenant = "smoke"
	const collection = "grpc_docs"

	if mode == "exercise" {
		_, err = client.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
			TenantId: tenant,
			Name:     collection,
			Fields: []*deepdatav3.VectorFieldConfig{{
				Name: "embedding", Type: 0, Dim: 3, IndexType: "flat",
			}},
		})
		must(err)
		_, err = client.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: tenant})
		must(err)
		listed, err := client.ListCollections(ctx, &deepdatav3.ListCollectionsRequest{TenantId: tenant})
		must(err)
		if len(listed.GetCollections()) < 2 {
			panic("gRPC list did not include HTTP and gRPC collections")
		}
		_, err = client.GetCollection(ctx, &deepdatav3.GetCollectionRequest{TenantId: tenant, Name: collection})
		must(err)
		inserted, err := client.Insert(ctx, &deepdatav3.InsertRequest{
			TenantId: tenant, Collection: collection, Id: 201,
			Vectors: map[string]*deepdatav3.VectorData{"embedding": dense(1, 0, 0)},
		})
		must(err)
		if inserted.GetId() != 201 {
			panic("gRPC insert returned the wrong ID")
		}
		batch, err := client.BatchInsert(ctx, &deepdatav3.BatchInsertRequest{
			TenantId:   tenant,
			Collection: collection,
			Docs: []*deepdatav3.BatchDoc{
				{Id: 202, Vectors: map[string]*deepdatav3.VectorData{"embedding": dense(0, 1, 0)}},
				{Id: 203, Vectors: map[string]*deepdatav3.VectorData{"embedding": dense(0, 0, 1)}},
			},
		})
		must(err)
		if batch.GetInserted() != 2 || len(batch.GetIds()) != 2 {
			panic("gRPC batch acknowledgement mismatch")
		}
		search, err := client.Search(ctx, &deepdatav3.SearchRequest{
			TenantId: tenant, Collection: collection, TopK: 10,
			Queries: map[string]*deepdatav3.VectorData{"embedding": dense(0, 1, 0)},
		})
		must(err)
		if !containsID(search.GetResults(), 202) {
			panic("gRPC search did not return the inserted document")
		}
		_, err = client.DeleteDoc(ctx, &deepdatav3.DeleteDocRequest{
			TenantId: tenant, Collection: collection, DocId: 201,
		})
		must(err)
		fmt.Println("gRPC exercise passed")
		return
	}

	if mode != "verify" {
		panic("unknown mode")
	}
	_, err = client.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: tenant})
	must(err)
	_, err = client.GetCollection(ctx, &deepdatav3.GetCollectionRequest{TenantId: tenant, Name: collection})
	must(err)
	search, err := client.Search(ctx, &deepdatav3.SearchRequest{
		TenantId: tenant, Collection: collection, TopK: 10,
		Queries: map[string]*deepdatav3.VectorData{"embedding": dense(0, 1, 0)},
	})
	must(err)
	if containsID(search.GetResults(), 201) || !containsID(search.GetResults(), 202) || !containsID(search.GetResults(), 203) {
		panic("gRPC restart state mismatch")
	}
	_, err = client.DeleteCollection(ctx, &deepdatav3.DeleteCollectionRequest{
		TenantId: tenant, Name: collection,
	})
	must(err)
	listed, err := client.ListCollections(ctx, &deepdatav3.ListCollectionsRequest{TenantId: tenant})
	must(err)
	for _, collectionInfo := range listed.GetCollections() {
		if collectionInfo.GetName() == collection {
			panic("gRPC collection remained after deletion")
		}
	}
	fmt.Println("gRPC restart verification passed")
}
GOEOF

go build -trimpath -o "$GRPC_PROBE" "$GRPC_SOURCE"

start_server() {
  : >>"$SERVER_LOG"
  VECTORDB_MODE=local \
  VECTORDB_BASE_DIR="$STATE_ROOT" \
  VECTORDB_DATA_DIR=local \
  PORT="$HTTP_PORT" \
  GRPC_PORT="$GRPC_PORT_NUMBER" \
  API_TOKEN="$API_TOKEN" \
  REQUIRE_AUTH=1 \
  LOG_FORMAT=json \
  "$BINARY" serve >>"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  local ready=false
  for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      fail "server exited before readiness"
    fi
    if curl -fsS --max-time 1 "$BASE_URL/readyz" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 0.1
  done
  [[ "$ready" == true ]] || fail "server did not become ready"
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill -TERM "$SERVER_PID"
  for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      local wait_rc=0
      wait "$SERVER_PID" || wait_rc=$?
      SERVER_PID=""
      ((wait_rc == 0)) || fail "server exited with status $wait_rc during graceful shutdown"
      return 0
    fi
    sleep 0.1
  done
  fail "server did not stop after SIGTERM"
}

api() {
  local method=$1 path=$2 body=${3-}
  if [[ -n "$body" ]]; then
    curl -fsS --max-time 10 -X "$method" "$BASE_URL$path" \
      -H "Authorization: Bearer $API_TOKEN" \
      -H 'Content-Type: application/json' \
      --data-binary "$body"
  else
    curl -fsS --max-time 10 -X "$method" "$BASE_URL$path" \
      -H "Authorization: Bearer $API_TOKEN"
  fi
}

assert_json() {
  local name=$1 json=$2 filter=$3
  if ! jq -e "$filter" <<<"$json" >/dev/null; then
    echo "$json" >&2
    fail "$name"
  fi
  echo "PASS: $name"
}

assert_status() {
  local name=$1 expected=$2 method=$3 path=$4
  local actual
  actual=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' \
    -X "$method" "$BASE_URL$path")
  [[ "$actual" == "$expected" ]] || fail "$name: expected $expected, got $actual"
  echo "PASS: $name"
}

echo "starting canonical server"
start_server

assert_status "liveness" 200 GET /livez
assert_status "readiness" 200 GET /readyz
assert_status "root mutation is unavailable" 404 POST /insert
assert_status "V2 collections are unavailable" 404 GET /v2/collections
assert_status "legacy dashboard is unavailable" 404 GET /dashboard/

CREATE_BODY='{
  "name":"http_docs",
  "fields":[
    {"name":"embedding","type":"dense","dim":3,"index":{"type":"hnsw"}},
    {"name":"keywords","type":"sparse","dim":64,"index":{"type":"inverted"}}
  ]
}'
response=$(api POST "/v3/tenants/$TENANT/collections" "$CREATE_BODY")
assert_json "HTTP create collection" "$response" '.status == "success"'

response=$(api GET "/v3/tenants/$TENANT/collections")
assert_json "HTTP list collections" "$response" '.count == 1 and .collections[0].Name == "http_docs"'
response=$(api GET "/v3/tenants/$TENANT/collections/http_docs")
assert_json "HTTP get collection" "$response" '.collection.Name == "http_docs"'

response=$(api POST "/v3/tenants/$TENANT/collections/http_docs/docs" '{
  "id":101,
  "vectors":{
    "embedding":[1,0,0],
    "keywords":{"indices":[1,3],"values":[0.8,0.4],"dim":64}
  },
  "metadata":{"kind":"single"}
}')
assert_json "HTTP insert" "$response" '.id == 101'

response=$(api POST "/v3/tenants/$TENANT/collections/http_docs/docs/batch" '{
  "documents":[
    {"id":102,"vectors":{
      "embedding":[0,1,0],
      "keywords":{"indices":[2],"values":[1],"dim":64}
    }},
    {"id":103,"vectors":{
      "embedding":[0,0,1],
      "keywords":{"indices":[3],"values":[1],"dim":64}
    }}
  ]
}')
assert_json "HTTP atomic batch" "$response" '.inserted == 2 and .ids == [102,103]'

response=$(api POST "/v3/tenants/$TENANT/collections/http_docs/search" '{
  "queries":{"embedding":[1,0,0]},"top_k":10
}')
assert_json "HTTP dense search" "$response" '([.documents[].id] | index(101)) != null'

response=$(api POST "/v3/tenants/$TENANT/collections/http_docs/search" '{
  "queries":{
    "embedding":[1,0,0],
    "keywords":{"indices":[1,3],"values":[0.8,0.4],"dim":64}
  },
  "top_k":10,
  "hybrid_params":{"strategy":"weighted","weights":{"embedding":0.7,"keywords":0.3}}
}')
assert_json "HTTP hybrid search" "$response" '([.documents[].id] | index(101)) != null'

response=$(api DELETE "/v3/tenants/$TENANT/collections/http_docs/docs" '{"doc_id":103}')
assert_json "HTTP delete document" "$response" '.status == "success"'
response=$(api GET "/v3/tenants/$TENANT")
assert_json "HTTP tenant info" "$response" '.tenant_id == "smoke" and .total_documents == 2'

"$GRPC_PROBE" exercise "$GRPC_ADDRESS" "$API_TOKEN"

echo "restarting canonical server"
stop_server
start_server

response=$(api GET "/v3/tenants/$TENANT/collections/http_docs")
assert_json "HTTP collection survived restart" "$response" '.collection.DocCount == 2'
response=$(api POST "/v3/tenants/$TENANT/collections/http_docs/search" '{
  "queries":{"embedding":[0,1,0]},"top_k":10
}')
assert_json "HTTP mutations survived restart" "$response" \
  '([.documents[].id] | index(101)) != null and
   ([.documents[].id] | index(102)) != null and
   ([.documents[].id] | index(103)) == null'

"$GRPC_PROBE" verify "$GRPC_ADDRESS" "$API_TOKEN"

response=$(api DELETE "/v3/tenants/$TENANT/collections/http_docs")
assert_json "HTTP delete collection" "$response" '.status == "success"'
response=$(api GET "/v3/tenants/$TENANT/collections")
assert_json "all smoke collections removed" "$response" '.count == 0'

stop_server
echo "canonical RC smoke passed"
