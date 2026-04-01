"use client";

import { Icon } from "@/components/ui/icon";

interface PrecisionFiltersProps {
  role: string;
  onRoleChange: (v: string) => void;
  location: string;
  onLocationChange: (v: string) => void;
  count: number;
  onCountChange: (v: number) => void;
}

export function PrecisionFilters({
  role,
  onRoleChange,
  location,
  onLocationChange,
  count,
  onCountChange,
}: PrecisionFiltersProps) {
  return (
    <div className="bg-card p-6 border border-navy">
      <div className="flex items-center gap-2 mb-8">
        <Icon name="tune" size={16} />
        <h3 className="font-heading font-bold text-navy uppercase tracking-widest text-sm">
          What are you looking for?
        </h3>
      </div>

      <div className="space-y-8">
        {/* Role */}
        <div>
          <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
            Job Title or Role
          </label>
          <input
            type="text"
            value={role}
            onChange={(e) => onRoleChange(e.target.value)}
            placeholder="SOFTWARE ENGINEER"
            className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-[11px] py-3 px-1 text-navy uppercase tracking-widest placeholder-muted/40"
          />
        </div>

        {/* Location */}
        <div>
          <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
            Where?
          </label>
          <input
            type="text"
            value={location}
            onChange={(e) => onLocationChange(e.target.value)}
            placeholder="REMOTE"
            className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-[11px] py-3 px-1 text-navy uppercase tracking-widest placeholder-muted/40"
          />
        </div>

        {/* Count */}
        <div>
          <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
            How many results?
          </label>
          <input
            type="range"
            min={10}
            max={100}
            step={5}
            value={count}
            onChange={(e) => onCountChange(Number(e.target.value))}
            className="w-full h-[2px] bg-surface-alt appearance-none cursor-pointer accent-primary"
          />
          <div className="flex justify-between mt-4 text-[10px] font-mono text-navy">
            <span>10</span>
            <span className="font-bold underline decoration-[2px] underline-offset-4">
              {count}
            </span>
            <span>100</span>
          </div>
        </div>
      </div>
    </div>
  );
}
