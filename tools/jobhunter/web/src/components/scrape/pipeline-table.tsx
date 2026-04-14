"use client";

import { useState, useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { Icon } from "@/components/ui/icon";
import type { JobRow } from "@/lib/types";
import Link from "next/link";

interface PipelineTableProps {
  jobs: JobRow[];
  globalFilter: string;
  onGlobalFilterChange: (v: string) => void;
}

export function PipelineTable({
  jobs,
  globalFilter,
  onGlobalFilterChange,
}: PipelineTableProps) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "fit_score", desc: true },
  ]);

  const columns = useMemo<ColumnDef<JobRow>[]>(
    () => [
      {
        accessorKey: "fit_score",
        header: "Match_Score",
        size: 120,
        cell: ({ getValue }) => {
          const v = getValue() as number | null;
          if (v === null) return <span className="text-muted text-[10px]">—</span>;
          return (
            <div className="font-mono text-lg font-black text-navy">
              {v * 10}.00%
            </div>
          );
        },
      },
      {
        accessorKey: "title",
        header: "Role_Company",
        size: 250,
        cell: ({ row }) => (
          <div>
            <p className="font-bold text-xs text-navy uppercase tracking-widest">
              {row.original.title}
            </p>
            <p className="text-[10px] text-muted uppercase tracking-widest mt-1">
              {row.original.company || "Unknown"}
            </p>
          </div>
        ),
      },
      {
        accessorKey: "location",
        header: "Location_Meta",
        size: 180,
        cell: ({ getValue }) => (
          <span className="text-[10px] font-mono text-navy uppercase">
            {(getValue() as string) || "N/A"}
          </span>
        ),
      },
      {
        id: "actions",
        header: "Actions",
        size: 140,
        cell: ({ row }) => (
          <div className="flex justify-end gap-3">
            <Link
              href={`/tailor?id=${row.original.id}`}
              className="h-10 px-6 border border-navy bg-navy text-white text-[10px] font-bold uppercase tracking-widest flex items-center gap-2 hover:bg-primary transition-sharp"
              onClick={(e) => e.stopPropagation()}
            >
              Tailor
            </Link>
          </div>
        ),
      },
    ],
    []
  );

  const table = useReactTable({
    data: jobs,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 10 } },
  });

  const pageIndex = table.getState().pagination.pageIndex;
  const pageCount = table.getPageCount();

  return (
    <div className="bg-card border border-navy">
      {/* Header */}
      <div className="px-8 py-6 flex justify-between items-center border-b border-navy">
        <h3 className="font-heading font-bold text-navy uppercase tracking-widest">
          Recent Pipeline Matches
        </h3>
        <div className="flex items-center gap-4">
          <span className="text-[9px] uppercase tracking-widest text-muted">Sort:</span>
          <button className="text-[10px] font-bold text-navy uppercase tracking-widest flex items-center gap-1">
            Relevance <Icon name="arrow_downward" size={12} />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead className="bg-surface">
            {table.getHeaderGroups().map((hg) => (
              <tr
                key={hg.id}
                className="text-[9px] uppercase tracking-[0.2em] text-muted border-b border-navy"
              >
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    className={`px-8 py-4 font-bold cursor-pointer select-none hover:text-navy transition-sharp ${
                      header.id === "actions" ? "text-right" : ""
                    }`}
                    style={{ width: header.getSize() }}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center gap-1">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === "asc" && (
                        <Icon name="arrow_upward" size={10} />
                      )}
                      {header.column.getIsSorted() === "desc" && (
                        <Icon name="arrow_downward" size={10} />
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-border-light">
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className="group hover:bg-surface transition-sharp"
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-8 py-8">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {jobs.length === 0 && (
        <div className="px-8 py-16 text-center">
          <p className="text-muted text-[11px] uppercase tracking-widest">
            No pipeline matches yet. Configure sources and initiate a scrape cycle.
          </p>
        </div>
      )}

      {/* Pagination */}
      <div className="px-8 py-6 bg-surface border-t border-navy flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-navy" />
          <span className="text-[9px] uppercase tracking-[0.2em] text-navy font-mono font-bold">
            SYSTEM_STATUS: OPTIMIZED
          </span>
        </div>
        <div className="flex gap-8 items-center">
          <button
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            className="text-[9px] uppercase tracking-widest font-bold text-muted hover:text-navy transition-sharp disabled:opacity-30"
          >
            PREV
          </button>
          <div className="flex items-center gap-4">
            {Array.from({ length: Math.min(pageCount, 5) }, (_, i) => (
              <button
                key={i}
                onClick={() => table.setPageIndex(i)}
                className={`text-[10px] font-bold ${
                  i === pageIndex
                    ? "text-navy border-b-2 border-navy"
                    : "text-muted"
                }`}
              >
                {String(i + 1).padStart(2, "0")}
              </button>
            ))}
          </div>
          <button
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            className="text-[9px] uppercase tracking-widest font-bold text-muted hover:text-navy transition-sharp disabled:opacity-30"
          >
            NEXT
          </button>
        </div>
      </div>
    </div>
  );
}
