// Base path for all API calls — Next.js basePath only auto-prefixes Link/router, not fetch()
// Use the Next.js internal variable which is always set correctly from next.config.ts
const BASE =
  process.env.NEXT_PUBLIC_BASE_PATH ||
  process.env.__NEXT_ROUTER_BASEPATH ||
  "";

export function api(path: string): string {
  return `${BASE}${path}`;
}
