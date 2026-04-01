"use client";

import { Icon } from "./icon";

interface StatCardProps {
  icon: string;
  value: number;
  label: string;
  progress?: number;
  badge?: string;
  color?: "blue" | "green" | "amber" | "default";
}

const colorMap = {
  blue: "text-primary",
  green: "text-accent",
  amber: "text-warning",
  default: "text-navy",
};

export function StatCard({ icon, value, label, progress, badge, color = "default" }: StatCardProps) {
  const display = String(value).padStart(2, "0");

  return (
    <div className="bg-card p-6 group cursor-default">
      <div className="flex items-start justify-between mb-6">
        <div className={`w-10 h-10 bg-surface-alt flex items-center justify-center ${colorMap[color]}`}>
          <Icon name={icon} size={20} />
        </div>
        {badge && (
          <span className="font-mono text-[9px] font-semibold text-accent bg-accent/10 px-2 py-0.5">
            {badge}
          </span>
        )}
      </div>
      <div className="mb-4">
        <div className="text-4xl font-heading font-bold text-navy leading-none">
          {display}
        </div>
        <div className="text-[10px] text-muted font-medium uppercase tracking-wider mt-2">
          {label}
        </div>
      </div>
      {progress !== undefined && (
        <div className="h-1 w-full bg-surface-alt">
          <div
            className="h-full bg-primary transition-sharp"
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      )}
    </div>
  );
}
