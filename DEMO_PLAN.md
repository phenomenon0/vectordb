# AgentScope × VectorDB Demo Plan

Goal: show an end-to-end AgentScope conversation that queries the vectordb HTTP service via the Go client + QueryTool adapter. Keep it minimal and runnable from this repo without extra setup beyond starting the vectordb binary.

## Preconditions
- vectordb server running locally on `:8080` (or env `VECTORDB_BASE_URL`).
- Hash embedder is fine for the demo; ONNX optional.
- API token optional; if set, the client/tool must send `Authorization: Bearer <token>`.

## Demo Skeleton (code outline)
1) Start scheduler + run loop.
2) Create vectordb client: `client.New(baseURL, client.WithToken(token))`.
3) Register `QueryTool` with policy `{DefaultTimeout: 2s}`.
4) Register an LLM tool (fastest router preset) or a stub if no key.
5) Agent logic:
   - On `MsgUserInput`, call QueryTool with `{Query: msg.Text, TopK: 3, IncludeMeta: true}`.
   - On `MsgToolResult` (query), stitch docs into a prompt and call LLM.
   - On final LLM result, `MarkComplete`.
6) Boot conversation with a sample question.

## Minimal code drop (pseudo)
```go
cli := client.New(baseURL, client.WithToken(token))
qt := &client.QueryTool{Client: cli}
qtID := sched.Tools().Register(qt, core.ToolPolicy{DefaultTimeout: 2 * time.Second}, nil)

llmCfg, _ := router.FastestModel()
llmID := sched.Tools().Register(tools.NewLLMTool(llmCfg), core.DefaultToolPolicy, nil)

agent := core.NewFuncAgent("rag-demo", func(ctx *core.AgentContext, msg *core.Message) {
  switch msg.Type {
  case core.MsgUserInput:
    ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
      ToolID: qtID,
      Input:  &client.QueryRequest{Query: msg.Text, TopK: 3, IncludeMeta: true},
    })
  case core.MsgToolResult:
    qr, _ := msg.ToolRes.Output.(*client.QueryOutput)
    prompt := strings.Join(qr.Docs, "\n---\n")
    ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
      ToolID: llmID,
      Input: &tools.LLMRequest{
        System: "Concise.",
        Messages: []tools.LLMMessage{{Role: "user", Content: prompt}},
      },
    })
  case core.MsgText:
    ctx.Scheduler.ConvMgr().MarkComplete(ctx.ConvID)
  }
})
```

## How to run
```bash
# Terminal 1: start vectordb service
go run ./vectordb

# Terminal 2: run the demo (to be added, e.g., cmd/vdb_demo/main.go)
go run ./cmd/vdb_demo
```

## Nice-to-haves (not required for first demo)
- Insert seed docs before query (via client.BatchInsert) if the index is empty.
- Metrics: watch `/metrics` or add a Prom sink tool.
- Config: read base URL/token from env vars in the demo (`VECTORDB_BASE_URL`, `API_TOKEN`).
