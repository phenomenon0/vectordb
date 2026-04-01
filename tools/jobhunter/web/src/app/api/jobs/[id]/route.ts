import { NextRequest, NextResponse } from "next/server";
import { getJob } from "@/lib/db";
import { requireAuth } from "@/lib/auth-helpers";
import { getUserDataDir } from "@/lib/user-data";
import { readFileSync, existsSync } from "fs";
import { resolve } from "path";

export const dynamic = "force-dynamic";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const result = await requireAuth();
  if ("error" in result) return result.error;

  const { id } = await params;
  const job = getJob(result.userId, parseInt(id, 10));
  if (!job) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  let resumeContent: string | null = null;
  if (job.resume_path && existsSync(job.resume_path)) {
    // Validate path is within the user's data directory
    const userDir = getUserDataDir(result.userId);
    const resolvedPath = resolve(job.resume_path);
    if (resolvedPath.startsWith(userDir)) {
      try {
        resumeContent = readFileSync(resolvedPath, "utf-8");
      } catch (e) {
        console.error(`Failed to read resume at ${resolvedPath}:`, e);
      }
    }
  }

  return NextResponse.json({ ...job, resumeContent });
}
