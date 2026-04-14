"use client";

interface IconProps {
  name: string;
  size?: number;
  className?: string;
  fill?: boolean;
}

export function Icon({ name, size = 24, className = "", fill = false }: IconProps) {
  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={{
        fontSize: size,
        fontVariationSettings: `'FILL' ${fill ? 1 : 0}, 'wght' 300, 'GRAD' 0, 'opsz' ${size}`,
      }}
    >
      {name}
    </span>
  );
}
