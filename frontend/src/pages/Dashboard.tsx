import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import {
  Activity,
  ArrowRight,
  Cpu,
  FileJson,
  Film,
  Globe,
  Image as ImageIcon,
  KeyRound,
  MessageSquare,
  Paperclip,
  RefreshCw,
  Server,
  Shield,
  ShieldCheck,
} from "lucide-react"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { toast } from "sonner"
import { PageShell } from "../components/ui/page-shell"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { StatusBadge, type BadgeTone } from "../components/ui/badge"
import { Notice } from "../components/ui/field"
import { cn } from "@/lib/utils"

type StatusResponse = {
  accounts?: { valid?: number; rate_limited?: number; invalid?: number }
}

/** 首屏快捷任务：把最常用的四个动作从侧栏搬到页面上，减少一次导航。 */
const QUICK_TASKS = [
  {
    title: "管理账号池",
    description: "巡检、启停与个性化上游账号",
    path: "/accounts",
    icon: Activity,
  },
  {
    title: "接口测试",
    description: "用一轮对话验证分发链路",
    path: "/test",
    icon: MessageSquare,
  },
  {
    title: "图片生成",
    description: "按提示词与比例出图",
    path: "/images",
    icon: ImageIcon,
  },
  {
    title: "视频生成",
    description: "按比例与时长生成视频",
    path: "/videos",
    icon: Film,
  },
]

const ENDPOINT_GROUPS: {
  label: string
  hint: string
  items: {
    method: string
    path: string
    protocol: string
    tone: BadgeTone
    icon: typeof FileJson
  }[]
}[] = [
  {
    label: "对话协议",
    hint: "兼容 OpenAI / Anthropic / Gemini 三种调用习惯",
    items: [
      { method: "POST", path: "/v1/chat/completions", protocol: "OpenAI", tone: "success", icon: FileJson },
      { method: "POST", path: "/v1/messages", protocol: "Anthropic", tone: "info", icon: Cpu },
      { method: "POST", path: "/v1beta/models/{model}:generateContent", protocol: "Gemini", tone: "warning", icon: Globe },
    ],
  },
  {
    label: "媒体生成",
    hint: "图片与视频的独立生成入口",
    items: [
      { method: "POST", path: "/v1/images/generations", protocol: "Image Gen", tone: "accent", icon: ImageIcon },
      { method: "POST", path: "/v1/videos/generations", protocol: "Video Gen", tone: "accent", icon: Film },
    ],
  },
  {
    label: "文件与健康检查",
    hint: "附件上传与网关存活探针",
    items: [
      { method: "POST", path: "/v1/files", protocol: "Files", tone: "info", icon: Paperclip },
      { method: "GET", path: "/healthz", protocol: "健康检查", tone: "neutral", icon: Shield },
    ],
  },
]

export default function Dashboard() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [failed, setFailed] = useState(false)

  const loadStatus = () => {
    setLoading(true)
    return fetch(`${API_BASE}/api/admin/status`, { headers: getAuthHeader() })
      .then(res => {
        if (!res.ok) throw new Error("Unauthorized")
        return res.json()
      })
      .then(data => {
        setStatus(data)
        setFailed(false)
      })
      .catch(() => {
        setFailed(true)
        toast.error("状态获取失败，请在「系统设置」检查您的当前会话 Key。")
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    let alive = true

    fetch(`${API_BASE}/api/admin/status`, { headers: getAuthHeader() })
      .then(res => {
        if (!res.ok) throw new Error("Unauthorized")
        return res.json()
      })
      .then(data => {
        if (!alive) return
        setStatus(data)
        setFailed(false)
      })
      .catch(() => {
        if (!alive) return
        setFailed(true)
        toast.error("状态获取失败，请在「系统设置」检查您的当前会话 Key。")
      })
      .finally(() => {
        if (alive) setLoading(false)
      })

    return () => {
      alive = false
    }
  }, [])

  const valid = status?.accounts?.valid
  const totalKnown =
    (status?.accounts?.valid ?? 0) + (status?.accounts?.rate_limited ?? 0) + (status?.accounts?.invalid ?? 0)

  const metrics = [
    {
      label: "可用账号",
      value: valid,
      icon: Server,
      hint: "当前可正常调用的上游账号",
      accent: "text-emerald-600 dark:text-emerald-400",
    },
    {
      label: "限流账号",
      value: status?.accounts?.rate_limited,
      icon: Activity,
      hint: "被上游限流、暂不可用",
      accent: "text-amber-600 dark:text-amber-400",
    },
    {
      label: "失效账号",
      value: status?.accounts?.invalid,
      icon: Shield,
      hint: "认证失败或已封禁",
      accent: "text-red-600 dark:text-red-400",
    },
  ]

  return (
    <PageShell
      actions={
        <Button variant="outline" onClick={() => void loadStatus()} disabled={loading}>
          <RefreshCw className={cn("mr-2 h-4 w-4", loading && "animate-spin")} /> 刷新状态
        </Button>
      }
    >
      <div className="space-y-6">
        {failed && (
          <Notice tone="error">
            无法读取运行状态。请在「系统设置」中确认当前会话 Key 是否正确。
          </Notice>
        )}

        {/* 账号池健康：首屏最重要的一个数字 + 分段构成 */}
        <Card>
          <CardContent className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div className="min-w-0">
              <p className="text-sm font-medium text-muted-foreground">账号池健康</p>
              <p className="tabular mt-2 text-4xl font-semibold tracking-tight text-foreground">
                {loading ? <span className="text-muted-foreground/40">—</span> : (valid ?? 0)}
                <span className="ml-2 text-base font-normal text-muted-foreground">
                  / {totalKnown} 个账号可用
                </span>
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                可用账号数直接决定网关的并发上限，账号失效时请及时巡检。
              </p>
            </div>
            <div className="flex shrink-0 flex-wrap gap-2">
              <Button asChild variant="outline">
                <Link to="/accounts">
                  <Activity className="mr-2 h-4 w-4" /> 进入账号池
                </Link>
              </Button>
              <Button asChild>
                <Link to="/test">
                  <ShieldCheck className="mr-2 h-4 w-4" /> 验证分发链路
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {metrics.map(metric => (
            <Card key={metric.label}>
              <CardContent className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-muted-foreground">{metric.label}</p>
                  <p
                    className={cn(
                      "tabular mt-2 text-3xl font-semibold tracking-tight",
                      loading ? "text-muted-foreground/40" : metric.accent,
                    )}
                  >
                    {loading ? "—" : (metric.value ?? 0)}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">{metric.hint}</p>
                </div>
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted">
                  <metric.icon className="h-4 w-4 text-muted-foreground" />
                </span>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* 快捷任务：常用入口前置，不必先翻菜单 */}
        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-foreground">快捷任务</h3>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {QUICK_TASKS.map(task => (
              <Link
                key={task.path}
                to={task.path}
                className="group rounded-xl border border-border bg-card p-4 transition-colors hover:border-primary/40 hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
                    <task.icon className="h-4 w-4 text-primary" />
                  </span>
                  <ArrowRight className="h-4 w-4 text-muted-foreground/50 transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
                </div>
                <p className="mt-3 text-sm font-medium text-foreground">{task.title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{task.description}</p>
              </Link>
            ))}
          </div>
        </section>

        {/* 接口按协议族归组，避免一长条平铺列表 */}
        <section className="space-y-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold text-foreground">开放接口</h3>
            <Link
              to="/settings"
              className="inline-flex items-center gap-1 text-xs text-primary transition-colors hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            >
              查看调用示例 <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            {ENDPOINT_GROUPS.map(group => (
              <Card key={group.label} className="overflow-hidden">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">{group.label}</CardTitle>
                  <CardDescription className="text-xs">{group.hint}</CardDescription>
                </CardHeader>
                <div className="divide-y divide-border">
                  {group.items.map(endpoint => (
                    <div
                      key={`${endpoint.method} ${endpoint.path}`}
                      className="flex items-center justify-between gap-3 px-5 py-3"
                    >
                      <div className="flex min-w-0 items-center gap-2.5">
                        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted">
                          <endpoint.icon className="h-3.5 w-3.5 text-muted-foreground" />
                        </span>
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] font-semibold text-muted-foreground">
                              {endpoint.method}
                            </span>
                            <span className="truncate font-mono text-xs text-foreground">
                              {endpoint.path}
                            </span>
                          </div>
                        </div>
                      </div>
                      <StatusBadge tone={endpoint.tone} className="shrink-0 text-[11px]">
                        {endpoint.protocol}
                      </StatusBadge>
                    </div>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        </section>

        <Card>
          <CardContent className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted">
                <KeyRound className="h-4 w-4 text-muted-foreground" />
              </span>
              <div>
                <p className="text-sm font-medium text-foreground">下游接入凭证</p>
                <p className="text-xs text-muted-foreground">
                  为客户端发放独立 API Key，避免直接共享控制台会话密钥。
                </p>
              </div>
            </div>
            <Button asChild variant="outline">
              <Link to="/tokens">
                管理 API Key <ArrowRight className="ml-1 h-3.5 w-3.5" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </PageShell>
  )
}
