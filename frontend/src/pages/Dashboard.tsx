import { useEffect, useState } from "react"
import { Server, ActivityIcon, FileJson, Cpu, Shield, Globe, ImageIcon, Paperclip } from "lucide-react"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { toast } from "sonner"
import { PageHeader } from "../components/ui/page-header"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/card"
import { StatusBadge, type BadgeTone } from "../components/ui/badge"
import { Notice } from "../components/ui/field"

type StatusResponse = {
  accounts?: { valid?: number; rate_limited?: number; invalid?: number }
}

const ENDPOINTS: {
  method: string
  path: string
  protocol: string
  tone: BadgeTone
  icon: typeof FileJson
}[] = [
  { method: "POST", path: "/v1/chat/completions", protocol: "OpenAI", tone: "success", icon: FileJson },
  { method: "POST", path: "/v1/messages", protocol: "Anthropic", tone: "info", icon: Cpu },
  { method: "POST", path: "/v1/models/gemini-pro:generateContent", protocol: "Gemini", tone: "warning", icon: Globe },
  { method: "POST", path: "/v1/images/generations", protocol: "Image Gen", tone: "accent", icon: ImageIcon },
  { method: "POST", path: "/v1/files", protocol: "Files", tone: "info", icon: Paperclip },
  { method: "GET", path: "/", protocol: "健康检查", tone: "neutral", icon: Shield },
]

export default function Dashboard() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [failed, setFailed] = useState(false)

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

  const metrics = [
    {
      label: "可用账号",
      value: status?.accounts?.valid,
      icon: Server,
      hint: "当前可正常调用的上游账号",
    },
    {
      label: "限流账号",
      value: status?.accounts?.rate_limited,
      icon: ActivityIcon,
      hint: "被上游限流、暂不可用",
    },
    {
      label: "失效账号",
      value: status?.accounts?.invalid,
      icon: Shield,
      hint: "认证失败或已封禁",
    },
  ]

  return (
    <div className="w-full space-y-6">
      <PageHeader
        title="运行状态"
        description="全局并发监控与千问账号池概览。"
      />

      {failed && (
        <Notice tone="error">
          无法读取运行状态。请在「系统设置」中确认当前会话 Key 是否正确。
        </Notice>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {metrics.map(metric => (
          <Card key={metric.label}>
            <CardContent className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium text-muted-foreground">{metric.label}</p>
                <p className="tabular mt-2 text-3xl font-semibold tracking-tight text-foreground">
                  {loading ? <span className="text-muted-foreground/40">—</span> : (metric.value ?? 0)}
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

      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle>API 接口池</CardTitle>
          <CardDescription>
            兼容主流 AI 协议的调用入口，默认无需认证，或通过 API Key 访问。
          </CardDescription>
        </CardHeader>

        <div className="divide-y divide-border">
          {ENDPOINTS.map(endpoint => (
            <div
              key={`${endpoint.method} ${endpoint.path}`}
              className="flex flex-col gap-2 px-5 py-3.5 transition-colors hover:bg-muted/40 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="flex min-w-0 items-center gap-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted">
                  <endpoint.icon className="h-4 w-4 text-muted-foreground" />
                </span>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] font-semibold text-muted-foreground">
                      {endpoint.method}
                    </span>
                    <span className="truncate font-mono text-sm text-foreground">
                      {endpoint.path}
                    </span>
                  </div>
                </div>
              </div>
              <StatusBadge tone={endpoint.tone} className="self-start sm:self-auto">
                {endpoint.protocol}
              </StatusBadge>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
