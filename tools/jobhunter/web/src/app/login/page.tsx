import { signIn } from "@/auth";
import { redirect } from "next/navigation";
import { auth } from "@/auth";

export default async function LoginPage() {
  const session = await auth();
  if (session?.user) {
    redirect("/");
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface p-6">
      <div className="w-full max-w-sm text-center">
        {/* Logo */}
        <div className="mb-10">
          <h1 className="font-heading text-3xl font-bold text-navy mb-2">
            Bounty Hunter
          </h1>
          <p className="text-[12px] text-muted">
            AI-powered job search companion
          </p>
        </div>

        {/* Login card */}
        <div className="bg-card border border-border-light p-8 shadow-sm">
          <h2 className="font-heading text-lg font-semibold text-navy mb-2">
            Welcome
          </h2>
          <p className="text-[12px] text-muted mb-8">
            Sign in to access your personalized job search dashboard
          </p>

          <form
            action={async () => {
              "use server";
              await signIn("google", { redirectTo: "/" });
            }}
          >
            <button
              type="submit"
              className="w-full flex items-center justify-center gap-3 bg-card border border-border hover:border-primary hover:bg-surface-alt transition-all duration-200 px-6 py-3 cursor-pointer"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              <span className="text-[13px] font-semibold text-navy">
                Continue with Google
              </span>
            </button>
          </form>

          <p className="text-[10px] text-muted mt-6">
            Your data is private and stored securely per account
          </p>
        </div>

        <p className="text-[9px] text-muted mt-8">
          Bounty Hunter — Find, Score, Tailor, Track
        </p>
      </div>
    </div>
  );
}
