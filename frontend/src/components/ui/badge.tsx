import * as React from "react"
import { cn } from "@/lib/utils"

export type BadgeTone = "neutral" | "success" | "warning" | "danger" | "info" | "accent"

const toneClasses: Record<BadgeTone, string> = {
  neutral: "bg-muted text-muted-foreground ring-border",
  success:
    "bg-emerald-500/10 text-emerald-700 ring-emerald-500/25 dark:text-emerald-300",
  warning: "bg-amber-500/10 text-amber-700 ring-amber-500/25 dark:text-amber-300",
  danger: "bg-red-500/10 text-red-700 ring-red-500/25 dark:text-red-300",
  info: "bg-blue-500/10 text-blue-700 ring-blue-500/25 dark:text-blue-300",
  accent: "bg-primary/10 text-primary ring-primary/25",
}

/** Status pill. Always carries a text label so state never depends on color alone. */
export function StatusBadge({
  tone = "neutral",
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset",
        toneClasses[tone],
        className,
      )}
      {...props}
    >
      {children}
    </span>
  )
}
