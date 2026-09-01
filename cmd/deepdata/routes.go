package main

import (
	"fmt"
	"io"

	"github.com/phenomenon0/vectordb/api/contract"
)

// routesHeader is the first TSV line printed by `deepdata routes`; the docs
// linter renders it as the header row of the generated table in
// internal/collection/API.md.
var routesHeader = []string{"method", "path", "permission", "grpc_rpc"}

// printRoutes writes the contract's HTTP surface as TSV: one line per
// operation in api/contract/v3/operations.json, columns method, path,
// permission and the gRPC method answering the same call (empty when the
// operation has no gRPC projection).
func printRoutes(w io.Writer) error {
	ops, err := contract.Operations()
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", routesHeader[0], routesHeader[1], routesHeader[2], routesHeader[3]); err != nil {
		return err
	}
	for _, op := range ops {
		if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", op.Method, op.Path, op.Permission, op.GRPCRPC); err != nil {
			return err
		}
	}
	return nil
}
