import * as React from "react"
import { useLocation } from "react-router-dom"
import { cn } from "@/lib/utils"
import { routeMeta } from "@/layouts/navigation"
import { PageHeaderBand } from "./page-header"

/**
 * 所有受管页面的统一骨架：粘性页面头（面包屑 + 标题 + 主操作）与定宽内容区。
 * 标题与说明默认从路由表推导，页面只需传入自己的主操作。
 */
export function PageShell({
  actions,
  children,
  title,
  description,
  group,
  contentClassName,
}: {
  actions?: React.ReactNode
  children: React.ReactNode
  title?: string
  description?: React.ReactNode
  group?: string
  contentClassName?: string
}) {
  const loc = useLocation()
  const meta = routeMeta(loc.pathname)

  return (
    <>
      <PageHeaderBand
        group={group ?? meta?.group}
        title={title ?? meta?.name ?? "控制台"}
        description={description ?? meta?.description}
        actions={actions}
      />
      <div
        className={cn(
          "mx-auto w-full max-w-7xl px-4 py-5 sm:px-6 md:py-6",
          contentClassName,
        )}
      >
        {children}
      </div>
    </>
  )
}
