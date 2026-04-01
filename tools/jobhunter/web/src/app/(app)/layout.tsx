"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { MobileNav } from "@/components/layout/mobile-nav";
import { Onboarding } from "@/components/onboarding";
import { api } from "@/lib/api";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const [onboarded, setOnboarded] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(api("/api/config"))
      .then((r) => r.json())
      .then((cfg) => setOnboarded(cfg.isOnboarded))
      .catch(() => setOnboarded(false));
  }, []);

  // Loading state
  if (onboarded === null) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-surface">
        <div className="text-center">
          <h1 className="font-heading text-2xl font-bold text-navy mb-4">
            Bounty Hunter
          </h1>
          <div className="flex items-center justify-center gap-2">
            <span className="w-1.5 h-3 bg-primary animate-pulse" />
            <span className="text-[11px] text-muted">
              Loading...
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Onboarding
  if (!onboarded) {
    return (
      <Onboarding
        onComplete={() => setOnboarded(true)}
      />
    );
  }

  // Main app shell
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 md:ml-60 min-h-screen bg-surface flex flex-col pb-16 md:pb-0">
        {children}
      </main>
      <MobileNav />
    </div>
  );
}
