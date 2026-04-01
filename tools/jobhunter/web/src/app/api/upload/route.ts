import { NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import { requireAuth } from "@/lib/auth-helpers";
import { getUserDataDir } from "@/lib/user-data";
import { join } from "path";

export async function POST(req: Request) {
  const result = await requireAuth();
  if ("error" in result) return result.error;

  try {
    const formData = await req.formData();
    const file = formData.get("resume") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    const name = file.name.toLowerCase();
    if (!name.endsWith(".pdf") && !name.endsWith(".md") && !name.endsWith(".txt")) {
      return NextResponse.json(
        { error: "Unsupported format. Upload PDF, MD, or TXT." },
        { status: 400 }
      );
    }

    const dataDir = getUserDataDir(result.userId);
    const ext = name.split(".").pop() || "pdf";
    const savePath = join(dataDir, `my_resume.${ext}`);
    const bytes = new Uint8Array(await file.arrayBuffer());
    await writeFile(savePath, bytes);

    return NextResponse.json({ success: true, path: savePath, name: file.name, size: file.size });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
