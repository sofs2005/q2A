import { Outlet, Link, useLocation } from "react-router-dom"
import { Menu, PanelLeftClose, PanelLeft, X } from "lucide-react"
import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { NAV_GROUPS, routeMeta } from "./navigation"

const COLLAPSE_STORAGE_KEY = "qwen2api_nav_collapsed"

export default function AdminLayout() {
  const loc = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)
  // 桌面端折叠为纯图标导航；偏好只存本地，与业务数据无关。
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(COLLAPSE_STORAGE_KEY) === "1",
  )

  const meta = routeMeta(loc.pathname)
  const activeName = meta?.name ?? "控制台"

  useEffect(() => {
    localStorage.setItem(COLLAPSE_STORAGE_KEY, collapsed ? "1" : "0")
  }, [collapsed])

  // Close the mobile drawer on Escape.
  useEffect(() => {
    if (!mobileOpen) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false)
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [mobileOpen])

  // 抽屉的关闭交给各个导航链接的 onClick 处理，避免在 effect 中同步改状态。
  return (
    <div className="flex min-h-screen w-full bg-background text-foreground transition-colors duration-200">
      {/* Mobile sidebar backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/30 backdrop-blur-[2px] md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        id="admin-sidebar"
        aria-label="主导航"
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-border bg-[hsl(var(--sidebar))] transition-[transform,width] duration-200 md:static md:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
          collapsed ? "md:w-[4.5rem]" : "md:w-64",
        )}
      >
        <div
          className={cn(
            "flex h-16 shrink-0 items-center border-b border-border",
            collapsed ? "md:justify-center md:px-2" : "justify-between px-5",
          )}
        >
          <Link
            to="/"
            className="flex items-center gap-2.5 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            title="返回运行状态"
          >
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">
              Q
            </span>
            <span className={cn("text-base font-semibold tracking-tight text-foreground", collapsed && "md:hidden")}>
              qwen2API
            </span>
          </Link>
          <button
            type="button"
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 md:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="关闭导航菜单"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 space-y-5 overflow-y-auto px-3 py-4">
          {NAV_GROUPS.map(group => (
            <div key={group.label} className="space-y-1">
              <p
                className={cn(
                  "px-3 pb-1 text-xs font-medium text-muted-foreground/80",
                  collapsed && "md:sr-only",
                )}
              >
                {group.label}
              </p>
              {group.items.map(item => {
                const active = loc.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    aria-current={active ? "page" : undefined}
                    title={collapsed ? item.name : undefined}
                    onClick={() => setMobileOpen(false)}
                    className={cn(
                      "relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                      collapsed && "md:justify-center md:px-2",
                      active
                        ? "bg-primary/10 font-medium text-primary"
                        : "font-normal text-muted-foreground hover:bg-accent hover:text-foreground",
                    )}
                  >
                    {active && (
                      <span
                        className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-primary"
                        aria-hidden="true"
                      />
                    )}
                    <item.icon className="h-4 w-4 shrink-0" />
                    <span className={cn("truncate", collapsed && "md:sr-only")}>{item.name}</span>
                  </Link>
                )
              })}
            </div>
          ))}
        </nav>

        <div className="shrink-0 border-t border-border p-3">
          <button
            type="button"
            onClick={() => setCollapsed(prev => !prev)}
            aria-expanded={!collapsed}
            aria-controls="admin-sidebar"
            className={cn(
              "hidden w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 md:flex",
              collapsed && "md:justify-center md:px-2",
            )}
            title={collapsed ? "展开导航" : "收起导航"}
          >
            {collapsed ? <PanelLeft className="h-4 w-4 shrink-0" /> : <PanelLeftClose className="h-4 w-4 shrink-0" />}
            <span className={cn(collapsed && "md:sr-only")}>收起导航</span>
          </button>
          <p className={cn("px-3 pt-2 text-xs text-muted-foreground", collapsed && "md:sr-only")}>
            本地网关控制台
          </p>
        </div>
      </aside>

      <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="z-30 flex h-16 shrink-0 items-center justify-between gap-4 border-b border-border bg-background/95 px-4 backdrop-blur md:hidden">
          <button
            type="button"
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            onClick={() => setMobileOpen(true)}
            aria-label="打开导航菜单"
            aria-controls="admin-sidebar"
            aria-expanded={mobileOpen}
          >
            <Menu className="h-5 w-5" />
          </button>
          <span className="truncate text-sm font-medium">{activeName}</span>
          <Link
            to="/settings"
            className="shrink-0 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          >
            设置
          </Link>
        </header>

        <div className="z-0 min-w-0 flex-1 overflow-y-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
