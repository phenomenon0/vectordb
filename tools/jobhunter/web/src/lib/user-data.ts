/**
 * Per-user data isolation.
 * Each authenticated user gets their own directory:
 *   ~/.local/share/jobhunter/users/{userId}/
 *     - jobs.db
 *     - profile.json
 *     - config.toml
 *     - resumes/
 */

import { join } from "path";
import { homedir } from "os";
import { mkdirSync, existsSync } from "fs";

const BASE_DATA_DIR = join(homedir(), ".local", "share", "jobhunter");

export function getUserDataDir(userId: string): string {
  const dir = join(BASE_DATA_DIR, "users", userId);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  return dir;
}

export function getUserDbPath(userId: string): string {
  return join(getUserDataDir(userId), "jobs.db");
}

export function getUserProfilePath(userId: string): string {
  return join(getUserDataDir(userId), "profile.json");
}

export function getUserConfigDir(userId: string): string {
  const dir = join(getUserDataDir(userId), "config");
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  return dir;
}

export function getUserConfigPath(userId: string): string {
  return join(getUserConfigDir(userId), "config.toml");
}

export function getUserResumesDir(userId: string): string {
  const dir = join(getUserDataDir(userId), "resumes");
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  return dir;
}

export function getUserEnvPath(userId: string): string {
  return join(getUserDataDir(userId), ".env");
}
