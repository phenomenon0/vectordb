"use client";

import { type ButtonHTMLAttributes, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "accent";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

const styles: Record<Variant, string> = {
  primary:
    "bg-navy text-white font-semibold text-[11px] tracking-wide uppercase px-6 py-3 border border-navy hover:bg-primary transition-sharp disabled:opacity-40 cursor-pointer",
  secondary:
    "bg-card text-navy font-semibold text-[11px] tracking-wide uppercase px-6 py-3 border border-border hover:border-primary hover:text-primary transition-sharp disabled:opacity-40 cursor-pointer",
  accent:
    "bg-accent text-white font-semibold text-[11px] tracking-wide uppercase px-6 py-3 border border-accent hover:bg-accent-dark transition-sharp disabled:opacity-40 cursor-pointer",
  ghost:
    "bg-transparent text-text-light font-medium text-[11px] tracking-wide uppercase px-4 py-2 hover:text-primary hover:bg-surface-alt transition-sharp disabled:opacity-40 cursor-pointer",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", className = "", children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={`${styles[variant]} ${className}`}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";
