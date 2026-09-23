import { Outlet, Link, useLocation } from "react-router-dom"
import { Activity, Key, Settings, LayoutDashboard, MessageSquare, Menu, X, Image, Film } from "lucide-react"
import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

const NAV_GROUPS = [
  {
    label: "概览",
    items: [{ name: "运行状态", path: "/", icon: LayoutDashboard }],
  },
  {
    label: "上游资源",
    items: [{ name: "账号管理", path: "/accounts", icon: Activity }],
  },
  {
    label: "API 与工具",
    items: [
      { name: "API Key", path: "/tokens", icon: Key },
      { name: "接口测试", path: "/test", icon: MessageSquare },
      { name: "图片生成", path: "/images", icon: Image },
      { name: "视频生成", path: "/videos", icon: Film },
    ],
  },
  {
    label: "系统",
    items: [{ name: "系统设置", path: "/settings", icon: Settings }],
  },
]

export default function AdminLayout() {
  const loc = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  const activeName =
    NAV_GROUPS.flatMap(group => group.items).find(item => item.path === loc.pathname)?.name ?? "qwen2API"

  // Close the mobile drawer on Escape.
  useEffect(() => {
    if (!mobileOpen) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false)
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [mobileOpen])

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
          "fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-border bg-[hsl(var(--sidebar))] transition-transform duration-200 md:static md:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-border px-5">
          <Link to="/" className="flex items-center gap-2.5 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">
              Q
            </span>
            <span className="text-base font-semibold tracking-tight text-foreground">qwen2API</span>
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

        <nav className="flex-1 space-y-6 overflow-y-auto px-3 py-4">
          {NAV_GROUPS.map(group => (
            <div key={group.label} className="space-y-1">
              <p className="px-3 pb-1 text-xs font-medium text-muted-foreground/80">{group.label}</p>
              {group.items.map(item => {
                const active = loc.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    aria-current={active ? "page" : undefined}
                    onClick={() => setMobileOpen(false)}
                    className={cn(
                      "relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
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
                    {item.name}
                  </Link>
                )
              })}
            </div>
          ))}
        </nav>

        <div className="border-t border-border px-5 py-3 text-xs text-muted-foreground">
          本地网关控制台
        </div>
      </aside>

      <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="z-10 flex h-16 items-center justify-between gap-4 border-b border-border bg-background/95 px-4 backdrop-blur md:hidden">
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
          <span className="h-8 w-8 shrink-0 rounded-lg bg-primary/10" aria-hidden="true" />
        </header>

        <div className="z-0 min-w-0 flex-1 overflow-y-auto p-4 sm:p-6 md:p-8">
          <div className="mx-auto w-full max-w-7xl">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  )
}
