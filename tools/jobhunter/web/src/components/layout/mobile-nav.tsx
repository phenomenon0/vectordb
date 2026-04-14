"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Icon } from "../ui/icon";

const ITEMS = [
  { href: "/", icon: "grid_view", label: "Home" },
  { href: "/scrape", icon: "search", label: "Find" },
  { href: "/tailor", icon: "auto_fix_high", label: "Tailor" },
  { href: "/tracker", icon: "assignment", label: "Track" },
  { href: "/profile", icon: "person", label: "Profile" },
];

export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-surface flex justify-around items-center h-16 border-t border-navy px-4 z-50">
      {ITEMS.map((item) => {
        const isActive =
          item.href === "/"
            ? pathname === "/"
            : pathname.startsWith(item.href);

        return (
          <Link
            key={item.href}
            href={item.href}
            className={`flex flex-col items-center gap-1 ${
              isActive ? "text-navy" : "text-muted"
            }`}
          >
            <Icon name={item.icon} size={18} />
            <span className="text-[8px] font-bold uppercase tracking-widest">
              {item.label}
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
