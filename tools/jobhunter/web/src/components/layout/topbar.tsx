"use client";

import { Icon } from "../ui/icon";

interface TopbarProps {
  title: string;
  subtitle?: string;
}

export function Topbar({ title, subtitle }: TopbarProps) {
  return (
    <header className="bg-card text-navy border-b border-border-light flex justify-between items-center w-full px-6 h-14 sticky top-0 z-40">
      <div className="flex items-center gap-3">
        <h2 className="font-heading text-lg font-semibold tracking-tight">
          {title}
        </h2>
        {subtitle && (
          <span className="text-[10px] text-muted font-medium hidden lg:block">
            — {subtitle}
          </span>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div className="relative hidden lg:block">
          <Icon name="search" size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
          <input
            type="text"
            placeholder="Search..."
            className="bg-surface border border-border-light text-[12px] pl-9 pr-4 py-1.5 w-48 focus:border-primary text-text placeholder-muted"
          />
        </div>
        <button className="text-muted hover:text-primary transition-sharp p-1 cursor-pointer">
          <Icon name="notifications" size={20} />
        </button>
        <button className="text-muted hover:text-primary transition-sharp p-1 cursor-pointer">
          <Icon name="help_outline" size={20} />
        </button>
      </div>
    </header>
  );
}
