import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  basePath: `${process.env.NEXT_PUBLIC_BASE_PATH || process.env.__NEXT_ROUTER_BASEPATH || ""}/api/auth`,
  pages: {
    signIn: "/login",
  },
  callbacks: {
    authorized({ auth }) {
      return !!auth?.user;
    },
    jwt({ token, account, profile }) {
      if (account && profile) {
        // Use Google's sub (stable unique ID) as our user ID
        token.userId = profile.sub;
      }
      return token;
    },
    session({ session, token }) {
      if (session.user && token.userId) {
        (session.user as unknown as Record<string, unknown>).id = token.userId;
      }
      return session;
    },
  },
  trustHost: true,
});
