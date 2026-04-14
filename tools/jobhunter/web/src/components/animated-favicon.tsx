"use client";

import { useEffect } from "react";

/**
 * Animated favicon — blinking terminal cursor.
 * Swaps between two SVG data-URI frames via a <link rel="icon"> element.
 * Frame 1: terminal prompt with cursor visible
 * Frame 2: terminal prompt with cursor hidden (blink off)
 */

const SIZE = 32;
const BG = "#0C4A6E"; // navy
const CURSOR = "#22C55E"; // accent green
const TEXT = "#22C55E";

function makeSVG(cursorOn: boolean): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${SIZE} ${SIZE}">
    <rect width="${SIZE}" height="${SIZE}" rx="4" fill="${BG}"/>
    <text x="5" y="22" font-family="monospace" font-size="16" font-weight="bold" fill="${TEXT}">&gt;_</text>
    ${cursorOn ? `<rect x="22" y="10" width="3" height="14" fill="${CURSOR}"/>` : ""}
  </svg>`;
}

function toDataURI(svg: string): string {
  return "data:image/svg+xml," + encodeURIComponent(svg);
}

const FRAMES = [toDataURI(makeSVG(true)), toDataURI(makeSVG(false))];

export function AnimatedFavicon() {
  useEffect(() => {
    let frame = 0;
    let link = document.querySelector<HTMLLinkElement>('link[rel="icon"]');

    if (!link) {
      link = document.createElement("link");
      link.rel = "icon";
      link.type = "image/svg+xml";
      document.head.appendChild(link);
    }

    link.href = FRAMES[0];

    const interval = setInterval(() => {
      frame = frame === 0 ? 1 : 0;
      link!.href = FRAMES[frame];
    }, 530);

    return () => clearInterval(interval);
  }, []);

  return null;
}
