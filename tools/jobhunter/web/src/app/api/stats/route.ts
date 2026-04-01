import { NextResponse } from "next/server";
import { getStats } from "@/lib/db";
import { requireAuth } from "@/lib/auth-helpers";

export const dynamic = "force-dynamic";

export async function GET() {
  const result = await requireAuth();
  if ("error" in result) return result.error;

  return NextResponse.json(getStats(result.userId));
}
