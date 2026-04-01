import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["better-sqlite3"],
  basePath: "/recruiter",
  output: "standalone",
};

export default nextConfig;
