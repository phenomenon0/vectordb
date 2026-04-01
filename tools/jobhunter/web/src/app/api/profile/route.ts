import { NextResponse } from "next/server";
import { readFile, writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { requireAuth } from "@/lib/auth-helpers";
import { getUserProfilePath, getUserDataDir } from "@/lib/user-data";

export const dynamic = "force-dynamic";

const DEFAULT_PROFILE = {
  name: "",
  headline: "",
  targetRoles: [],
  experienceLevel: "",
  locationPreference: "",
  salaryExpectation: "",
  skills: [],
  linkedinUrl: "",
  portfolioUrl: "",
  onboardedAt: null,
};

export async function GET() {
  const result = await requireAuth();
  if ("error" in result) return result.error;

  const profilePath = getUserProfilePath(result.userId);
  if (!existsSync(profilePath)) {
    return NextResponse.json(DEFAULT_PROFILE);
  }

  try {
    const raw = await readFile(profilePath, "utf-8");
    return NextResponse.json({ ...DEFAULT_PROFILE, ...JSON.parse(raw) });
  } catch {
    return NextResponse.json(DEFAULT_PROFILE);
  }
}

export async function POST(req: Request) {
  const result = await requireAuth();
  if ("error" in result) return result.error;

  const profilePath = getUserProfilePath(result.userId);
  const body = await req.json();

  let existing = { ...DEFAULT_PROFILE };
  if (existsSync(profilePath)) {
    try {
      existing = { ...DEFAULT_PROFILE, ...JSON.parse(await readFile(profilePath, "utf-8")) };
    } catch {}
  }

  const profile = {
    ...existing,
    ...body,
    onboardedAt: existing.onboardedAt || new Date().toISOString(),
  };

  await writeFile(profilePath, JSON.stringify(profile, null, 2), "utf-8");
  return NextResponse.json({ ok: true, profile });
}
