import { auth } from "@/auth";
import { NextResponse } from "next/server";

/**
 * Get the authenticated user's ID from the session.
 * Returns null if not authenticated or ID is invalid.
 */
export async function getUserId(): Promise<string | null> {
  const session = await auth();
  const user = session?.user as unknown as Record<string, unknown> | undefined;
  const id = user?.id;
  if (!id || typeof id !== "string" || id.length === 0 || id.length > 256) {
    return null;
  }
  return id;
}

/**
 * Require authentication — returns userId or 401 response.
 */
export async function requireAuth(): Promise<
  { userId: string } | { error: NextResponse }
> {
  const userId = await getUserId();
  if (!userId) {
    return {
      error: NextResponse.json({ error: "Unauthorized" }, { status: 401 }),
    };
  }
  return { userId };
}
