import * as React from "react"
import { Link } from "react-router-dom"
import { ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"

/** 面包屑：第一个节点恒为可点击的「控制台」根，末节点为当前页。 */
export function Breadcrumb({ group, title }: { group?: string; title: string }) {
  return (
    <nav aria-label="当前位置" className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
      <Link
        to="/"
        className="rounded transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
      >
        控制台
      </Link>
      {group ? (
        <>
          <ChevronRight className="h-3 w-3 shrink-0 opacity-60" aria-hidden="true" />
          <span className="truncate">{group}</span>
        </>
      ) : null}
      <ChevronRight className="h-3 w-3 shrink-0 opacity-60" aria-hidden="true" />
      <span aria-current="page" className="truncate font-medium text-foreground">
        {title}
      </span>
    </nav>
  )
}

/** 页面标题块：标题、一句话说明与可选操作区。 */
export function PageHeader({
  title,
  description,
  actions,
  className,
}: {
  title: string
  description?: React.ReactNode
  actions?: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-4 md:flex-row md:items-start md:justify-between",
        className,
      )}
    >
      <div className="min-w-0">
        <h2 className="text-2xl font-semibold tracking-tight text-foreground">{title}</h2>
        {description ? (
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  )
}

/**
 * 全局壳层的粘性页面头：面包屑 + 标题 + 该页的主操作。
 * 标题与主操作由壳层统一渲染，页面本身不再重复标题块，避免滚动后丢失上下文。
 */
export function PageHeaderBand({
  group,
  title,
  description,
  actions,
}: {
  group?: string
  title: string
  description?: React.ReactNode
  actions?: React.ReactNode
}) {
  return (
    <div className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-4 py-3 sm:px-6 md:flex-row md:items-center md:justify-between md:py-4">
        <div className="min-w-0 space-y-1">
          <Breadcrumb group={group} title={title} />
          <h1 className="truncate text-lg font-semibold tracking-tight text-foreground">{title}</h1>
          {description ? (
            <p className="hidden text-sm text-muted-foreground sm:block">{description}</p>
          ) : null}
        </div>
        {actions ? (
          <div className="flex flex-wrap items-center gap-2 md:shrink-0 md:justify-end">{actions}</div>
        ) : null}
      </div>
    </div>
  )
}
