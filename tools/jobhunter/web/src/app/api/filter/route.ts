import { NextResponse } from "next/server";
import { runCommand } from "@/lib/runner";
import { requireAuth } from "@/lib/auth-helpers";

export async function POST(req: Request) {
  const result = await requireAuth();
  if ("error" in result) return result.error;

  const body = await req.json();
  const { resume, model } = body as { resume?: string; model?: string };

  const args = ["filter"];
  if (resume) args.push("-r", resume);
  if (model) args.push("-m", model);

  const cmdResult = await runCommand(args);
  return NextResponse.json({
    success: cmdResult.exitCode === 0,
    output: cmdResult.stdout + cmdResult.stderr,
  });
}
