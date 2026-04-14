import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { randomUUID } from "crypto";

const COOKIE_NAME = "bh_uid";
const MAX_AGE = 60 * 60 * 24 * 365 * 2; // 2 years

export const dynamic = "force-dynamic";

/**
 * GET /api/session — returns current user ID, creating one if needed.
 * The UUID is stored in an httpOnly cookie that persists across sessions.
 */
export async function GET() {
  const cookieStore = await cookies();
  let uid = cookieStore.get(COOKIE_NAME)?.value;

  if (!uid) {
    uid = randomUUID();
    const res = NextResponse.json({ uid, isNew: true });
    res.cookies.set(COOKIE_NAME, uid, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: MAX_AGE,
      path: "/",
    });
    return res;
  }

  return NextResponse.json({ uid, isNew: false });
}
