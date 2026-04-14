import { cookies } from "next/headers";
import { NextResponse } from "next/server";

const COOKIE_NAME = "bh_uid";

/**
 * Get user ID from the bh_uid cookie.
 * Each browser gets a unique UUID on first visit — no login required.
 */
export async function getUserId(): Promise<string | null> {
  const cookieStore = await cookies();
  const uid = cookieStore.get(COOKIE_NAME)?.value;
  if (!uid || typeof uid !== "string" || uid.length < 8 || uid.length > 64) {
    return null;
  }
  return uid;
}

/**
 * Require a user ID — returns userId or 401.
 */
export async function requireAuth(): Promise<
  { userId: string } | { error: NextResponse }
> {
  const userId = await getUserId();
  if (!userId) {
    return {
      error: NextResponse.json(
        { error: "No user session. Clear cookies and reload." },
        { status: 401 }
      ),
    };
  }
  return { userId };
}
