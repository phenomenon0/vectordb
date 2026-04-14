import { NextResponse } from "next/server";
import { runCommand } from "@/lib/runner";
import { requireAuth } from "@/lib/auth-helpers";
import { getUserEnvPath } from "@/lib/user-data";

// Alphanumeric, spaces, hyphens, commas, periods only
const SAFE_TEXT = /^[a-zA-Z0-9\s\-,.'()\/+]+$/;
const VALID_SITES = new Set(["linkedin", "indeed", "glassdoor", "zip_recruiter", "google"]);

export async function POST(req: Request) {
  const result = await requireAuth();
  if ("error" in result) return result.error;

  const body = await req.json();
  const { role, location, count, sites } = body as {
    role: string; location?: string; count?: number; sites?: string;
  };

  // Validate role
  if (!role || typeof role !== "string" || role.length > 200 || !SAFE_TEXT.test(role)) {
    return NextResponse.json({ error: "Invalid role" }, { status: 400 });
  }

  // Validate location
  if (location && (typeof location !== "string" || location.length > 200 || !SAFE_TEXT.test(location))) {
    return NextResponse.json({ error: "Invalid location" }, { status: 400 });
  }

  // Validate count
  if (count !== undefined && (!Number.isInteger(count) || count < 1 || count > 200)) {
    return NextResponse.json({ error: "Count must be 1-200" }, { status: 400 });
  }

  // Validate sites
  if (sites) {
    const siteList = sites.split(",").map((s) => s.trim());
    if (siteList.some((s) => !VALID_SITES.has(s))) {
      return NextResponse.json({ error: `Invalid site. Allowed: ${[...VALID_SITES].join(", ")}` }, { status: 400 });
    }
  }

  const args = ["scrape", role];
  if (location) args.push("-l", location);
  if (count) args.push("-n", String(count));
  if (sites) args.push("--sites", sites);

  const cmdResult = await runCommand(args, undefined, getUserEnvPath(result.userId));
  return NextResponse.json({
    success: cmdResult.exitCode === 0,
    output: cmdResult.stdout + cmdResult.stderr,
  });
}
