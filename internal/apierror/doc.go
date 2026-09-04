// Package apierror is the one error vocabulary of the API. An engine error is
// classified once, here, into a code with a hint and a docs pointer; the HTTP,
// gRPC and MCP surfaces only project the same Error. Gate CTL-01.
//
// Tower layer: L3 transports.
package apierror
