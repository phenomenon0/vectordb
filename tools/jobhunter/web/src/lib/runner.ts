import { execFile } from "child_process";
import { readFileSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

const JOBHUNTER = "jobhunter";

const ALLOWED_ENV_KEYS = new Set(["ANTHROPIC_API_KEY"]);

/**
 * Load per-user env file, whitelisting only known-safe variables.
 */
function loadEnv(envPath?: string): Record<string, string> {
  const extra: Record<string, string> = {};
  const path = envPath || join(homedir(), ".local", "share", "jobhunter", ".env");
  if (existsSync(path)) {
    const lines = readFileSync(path, "utf-8").split("\n");
    for (const line of lines) {
      const match = line.match(/^([A-Z_]+)=(.+)$/);
      if (match && ALLOWED_ENV_KEYS.has(match[1])) {
        extra[match[1]] = match[2];
      }
    }
  }
  return extra;
}

/**
 * Run jobhunter CLI command safely using execFile (no shell injection).
 * Args are passed as an array directly — never interpolated into a string.
 */
export async function runCommand(
  args: string[],
  env?: Record<string, string>,
  envFilePath?: string
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const envVars = { ...process.env, ...loadEnv(envFilePath), ...env };

  return new Promise((resolve) => {
    execFile(
      JOBHUNTER,
      args,
      { timeout: 300_000, env: envVars, maxBuffer: 10 * 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) {
          resolve({
            stdout: stdout || "",
            stderr: stderr || String(err),
            exitCode: (err as NodeJS.ErrnoException).code ? 1 : 1,
          });
        } else {
          resolve({ stdout, stderr, exitCode: 0 });
        }
      }
    );
  });
}
