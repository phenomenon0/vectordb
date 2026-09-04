// Package collection is the DeepData engine: collections and their schemas,
// the hnsw/flat/inverted indexes behind them, fusion, metadata filters and the
// UsageTracker, together with the admission limits and the typed error
// sentinels the transports classify on. DurableStore adds the journal and
// snapshot persistence underneath, and TenantManager scopes both per tenant.
//
// Tower layers: L1 engine, and L0 durability via DurableStore.
package collection
