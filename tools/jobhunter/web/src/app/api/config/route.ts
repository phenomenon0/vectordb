import { NextResponse } from "next/server";
import { readFile, writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { requireAuth } from "@/lib/auth-helpers";
import {
  getUserConfigPath,
  getUserConfigDir,
  getUserDataDir,
  getUserProfilePath,
  getUserEnvPath,
} from "@/lib/user-data";

export const dynamic = "force-dynamic";

export async function GET() {
  const result = await requireAuth();
  if ("error" in result) return result.error;
  const { userId } = result;

  try {
    const CONFIG_FILE = getUserConfigPath(userId);
    const DATA_DIR = getUserDataDir(userId);
    const PROFILE_FILE = getUserProfilePath(userId);

    let resumePath: string | null = null;
    let model = "claude-haiku-4-5-20251001";
    let envPath = getUserEnvPath(userId);
    let hasApiKey = !!process.env.ANTHROPIC_API_KEY;
    let configExists = existsSync(CONFIG_FILE);

    // Check .env file for per-user API key
    if (!hasApiKey && existsSync(envPath)) {
      try {
        const envContent = await readFile(envPath, "utf-8");
        if (envContent.includes("ANTHROPIC_API_KEY=")) hasApiKey = true;
      } catch {}
    }

    if (configExists) {
      const raw = await readFile(CONFIG_FILE, "utf-8");
      const pathMatch = raw.match(/^path\s*=\s*"(.+)"/m);
      if (pathMatch) resumePath = pathMatch[1];
      const modelMatch = raw.match(/^model\s*=\s*"(.+)"/m);
      if (modelMatch) model = modelMatch[1];
    }

    let resumeExists = false;
    let resumeName: string | null = null;
    if (resumePath && existsSync(resumePath)) {
      resumeExists = true;
      resumeName = resumePath.split("/").pop() || null;
    }

    // Check for uploaded resumes in user data dir
    for (const ext of ["pdf", "md", "txt"]) {
      const p = `${DATA_DIR}/my_resume.${ext}`;
      if (!resumeExists && existsSync(p)) {
        resumePath = p;
        resumeExists = true;
        resumeName = `my_resume.${ext}`;
        break;
      }
    }

    // Onboarded if profile exists
    let profileOnboarded = false;
    if (existsSync(PROFILE_FILE)) {
      try {
        const profile = JSON.parse(await readFile(PROFILE_FILE, "utf-8"));
        profileOnboarded = !!profile.onboardedAt;
      } catch {}
    }
    const isOnboarded = profileOnboarded || resumeExists || configExists;

    return NextResponse.json({
      isOnboarded,
      resumePath,
      resumeExists,
      resumeName,
      model,
      hasApiKey,
      configExists,
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function POST(req: Request) {
  const result = await requireAuth();
  if ("error" in result) return result.error;
  const { userId } = result;

  try {
    const body = await req.json();
    const { model, apiKey } = body as { model?: string; apiKey?: string };

    const CONFIG_DIR = getUserConfigDir(userId);
    const DATA_DIR = getUserDataDir(userId);

    // Find resume
    let resumePath = "";
    for (const ext of ["pdf", "md", "txt"]) {
      const p = `${DATA_DIR}/my_resume.${ext}`;
      if (existsSync(p)) { resumePath = p; break; }
    }

    const config = `# JobHunter configuration

[llm]
model = "${model || "claude-haiku-4-5-20251001"}"
min_score = 6

[scrape]
sites = ["linkedin", "indeed"]
results_wanted = 50
country = "USA"

[resume]
path = "${resumePath}"
`;
    await writeFile(getUserConfigPath(userId), config, "utf-8");

    if (apiKey) {
      await writeFile(getUserEnvPath(userId), `ANTHROPIC_API_KEY=${apiKey}\n`, { mode: 0o600 });
    }

    return NextResponse.json({ success: true });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
