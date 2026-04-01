"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Icon } from "../ui/icon";
import { api } from "@/lib/api";

const NAV_ITEMS = [
  { href: "/", icon: "grid_view", label: "Dashboard" },
  { href: "/scrape", icon: "search", label: "Find Jobs" },
  { href: "/tailor", icon: "auto_fix_high", label: "Tailor" },
  { href: "/tracker", icon: "assignment", label: "Tracker" },
  { href: "/profile", icon: "person", label: "Profile" },
  { href: "/settings", icon: "settings", label: "Settings" },
];

interface SessionUser {
  name?: string;
  email?: string;
  image?: string;
}

export function Sidebar() {
  const pathname = usePathname();
  const [user, setUser] = useState<SessionUser | null>(null);
  const [profileName, setProfileName] = useState("");

  useEffect(() => {
    // Get session info
    fetch(api("/api/auth/session"))
      .then((r) => r.json())
      .then((s) => { if (s?.user) setUser(s.user); })
      .catch(() => {});
    // Get profile name
    fetch(api("/api/profile"))
      .then((r) => r.json())
      .then((p) => { if (p.name) setProfileName(p.name); })
      .catch(() => {});
  }, []);

  const displayName = profileName || user?.name || "You";
  const firstName = displayName.split(" ")[0];

  return (
    <aside className="hidden md:flex flex-col h-full w-60 fixed left-0 top-0 bg-card border-r border-border-light z-50">
      <div className="px-6 py-8">
        {/* Branding */}
        <div className="mb-8">
          <h1 className="font-heading text-xl font-bold text-navy tracking-tight">
            Bounty Hunter
          </h1>
          <span className="text-[10px] text-muted font-medium">
            Job Intelligence Platform
          </span>
        </div>

        {/* CTA */}
        <Link
          href="/scrape"
          className="w-full py-3 px-4 mb-8 bg-accent text-white font-semibold text-[11px] tracking-wide uppercase transition-sharp border border-accent hover:bg-accent-dark flex items-center justify-center gap-2 cursor-pointer"
        >
          <Icon name="add" size={16} className="text-white" />
          Find New Jobs
        </Link>

        {/* Nav */}
        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-2.5 transition-sharp cursor-pointer ${
                  isActive
                    ? "bg-primary/10 text-primary border-l-3 border-primary font-semibold"
                    : "text-text-light hover:bg-surface-alt hover:text-navy"
                }`}
              >
                <Icon name={item.icon} size={18} />
                <span className="text-[11px] font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* User footer */}
      <div className="mt-auto px-6 py-5 border-t border-border-light">
        <Link href="/profile" className="flex items-center gap-3 mb-3 group cursor-pointer">
          {user?.image ? (
            <img
              src={user.image}
              alt={displayName}
              className="w-9 h-9 object-cover"
              referrerPolicy="no-referrer"
            />
          ) : (
            <div className="w-9 h-9 bg-primary flex items-center justify-center">
              <span className="text-white font-heading font-bold text-sm">
                {firstName[0]?.toUpperCase() || "?"}
              </span>
            </div>
          )}
          <div className="min-w-0">
            <p className="text-[12px] font-semibold text-navy truncate group-hover:text-primary transition-sharp">
              {displayName}
            </p>
            <p className="text-[9px] text-muted truncate">
              {user?.email || "View profile"}
            </p>
          </div>
        </Link>
        <form action={api("/api/auth/signout")} method="POST">
          <button
            type="submit"
            className="flex items-center gap-2 text-[10px] text-muted hover:text-danger transition-sharp cursor-pointer"
          >
            <Icon name="logout" size={14} />
            Sign out
          </button>
        </form>
      </div>
    </aside>
  );
}
