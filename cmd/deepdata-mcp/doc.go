// Command deepdata-mcp exposes a running DeepData server to agents as six
// memory verbs over MCP (Model Context Protocol) on stdio.
//
// It is a self-contained JSON-RPC 2.0 implementation over newline-delimited
// JSON with zero external dependencies: it speaks the canonical tenant HTTP
// protocol, so it needs no engine build. Tool input/output schemas and the
// deepdata://contract resource are served verbatim from api/contract.
//
// Tower layer: L5 agent.
//
// Configuration (environment):
//
//	DEEPDATA_URL         base URL of the DeepData server (default http://127.0.0.1:8080)
//	DEEPDATA_TENANT      tenant id to operate on (default "mcp"); never a tool argument
//	DEEPDATA_COLLECTION  default collection for every verb (default "memory")
//	DEEPDATA_API_KEY     optional bearer token
//
// Exposed tools: deepdata_recall, deepdata_remember, deepdata_forget,
// deepdata_get, deepdata_collections, deepdata_create_collection.
package main
