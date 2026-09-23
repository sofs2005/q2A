import { useState, useEffect } from "react"
import { RefreshCw, KeyRound, ServerCrash, Code, Settings2, SlidersHorizontal, ArrowRight } from "lucide-react"
import { Button } from "../components/ui/button"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/card"
import { Input, Textarea, Field, Notice } from "../components/ui/field"
import { PageShell } from "../components/ui/page-shell"
import { cn } from "@/lib/utils"

type AdminSettings = {
  version?: string
  max_inflight_per_account?: number
  model_aliases?: Record<string, unknown>
}

type SectionKey = "auth" | "runtime" | "routing" | "developer"

const SECTIONS: { key: SectionKey; label: string; hint: string; icon: typeof KeyRound }[] = [
  { key: "auth", label: "控制台认证", hint: "本机浏览器的会话密钥与连接地址", icon: KeyRound },
  { key: "runtime", label: "网关运行", hint: "版本信息与单账号并发上限", icon: SlidersHorizontal },
  { key: "routing", label: "模型路由", hint: "下游模型名到实际模型的映射", icon: Settings2 },
  { key: "developer", label: "开发者接入", hint: "各协议的最小可用调用示例", icon: Code },
]

export default function SettingsPage() {
  const [settings, setSettings] = useState<AdminSettings | null>(null)
  const [sessionKey, setSessionKey] = useState(() => localStorage.getItem("qwen2api_key") || "")
  const [maxInflight, setMaxInflight] = useState(4)
  const [modelAliases, setModelAliases] = useState("")
  const [loadError, setLoadError] = useState(false)
  const [section, setSection] = useState<SectionKey>("auth")

  const fetchSettings = () => {
    fetch(`${API_BASE}/api/admin/settings`, { headers: getAuthHeader() })
      .then(res => {
        if(!res.ok) throw new Error("Unauthorized")
        return res.json()
      })
      .then(data => {
        setSettings(data)
        setMaxInflight(data.max_inflight_per_account || 4)
        setModelAliases(JSON.stringify(data.model_aliases || {}, null, 2))
        setLoadError(false)
      })
      .catch(() => {
        setLoadError(true)
        toast.error("配置获取失败，请检查会话 Key")
      })
  }

  useEffect(() => {
    fetchSettings()
  }, [])

  const handleSaveSessionKey = () => {
    if (!sessionKey.trim()) {
      toast.error("请输入 Key")
      return
    }
    localStorage.setItem('qwen2api_key', sessionKey.trim())
    toast.success("Key 已保存到本地，刷新数据...")
    fetchSettings()
  }

  const handleClearSessionKey = () => {
    localStorage.removeItem('qwen2api_key')
    setSessionKey("")
    toast.success("Key 已清除")
  }

  const handleSaveConcurrency = () => {
    fetch(`${API_BASE}/api/admin/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ max_inflight_per_account: Number(maxInflight) })
    }).then(res => {
      if(res.ok) { toast.success("并发配置已保存"); fetchSettings(); }
      else toast.error("保存失败")
    })
  }

  const handleSaveAliases = () => {
    try {
      const parsed = JSON.parse(modelAliases)
      fetch(`${API_BASE}/api/admin/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
        body: JSON.stringify({ model_aliases: parsed })
      }).then(res => {
        if(res.ok) { toast.success("模型映射规则已更新"); fetchSettings(); }
        else toast.error("保存失败")
      })
    } catch {
      toast.error("JSON 格式错误，请检查语法")
    }
  }

  const baseUrl = API_BASE || `http://${window.location.hostname}:7860`

  const curlExample = `# OpenAI streaming chat
  curl ${baseUrl}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d '{
      "model": "qwen3.6-plus",
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": true
    }'

  # Upload one file first (the response contains a reusable content_block)
  curl ${baseUrl}/v1/files \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -F "file=@./context.txt"

  # OpenAI + attachment
  curl ${baseUrl}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d '{
      "model": "qwen3.6-plus",
      "stream": false,
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Read the uploaded file and summarize the key points."},
            {"type": "input_file", "file_id": "FILE_ID_FROM_UPLOAD", "filename": "context.txt", "mime_type": "text/plain"}
          ]
        }
      ]
    }'

  # Anthropic / Claude Code + attachment
  curl ${baseUrl}/anthropic/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: YOUR_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
      "model": "claude-sonnet-4-6",
      "max_tokens": 1024,
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Read the uploaded file and summarize the key points."},
            {"type": "input_file", "file_id": "FILE_ID_FROM_UPLOAD", "filename": "context.txt", "mime_type": "text/plain"}
          ]
        }
      ]
    }'

  # Gemini
  curl ${baseUrl}/v1beta/models/qwen3.6-plus:generateContent \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d '{
      "contents": [{"parts": [{"text": "Hello"}]}]
    }'

  # Images
  curl ${baseUrl}/v1/images/generations \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d '{
      "model": "qwen3.8-max",
      "prompt": "A cyberpunk cat with neon lights, ultra realistic",
      "n": 1,
      "size": "1024x1024",
      "response_format": "url"
    }'

  # Video (reserved path)
  curl ${baseUrl}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d '{
      "model": "qwen3.6-plus",
      "stream": false,
      "messages": [{"role": "user", "content": "Generate a slow-motion ocean-wave video."}]
    }'`

  return (
    <PageShell
      actions={
        <Button variant="outline" onClick={() => {fetchSettings(); toast.success("配置已刷新")}}>
          <RefreshCw className="mr-2 h-4 w-4" /> 刷新配置
        </Button>
      }
    >
      <div className="space-y-6">
        {loadError && (
          <Notice tone="error">
            无法读取网关配置。请先在「控制台认证」中填入有效的 API Key 并保存。
          </Notice>
        )}

        {/* 桌面：左侧分区导航 + 右侧主面板；窄屏改为分区切换胶囊 + 堆叠 */}
        <div className="grid gap-6 lg:grid-cols-[16rem_minmax(0,1fr)] lg:items-start">
          <nav aria-label="设置分区" className="lg:sticky lg:top-[5.5rem]">
            <div className="flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:gap-1 lg:overflow-visible lg:pb-0">
              {SECTIONS.map(item => {
                const active = section === item.key
                return (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => setSection(item.key)}
                    aria-current={active ? "true" : undefined}
                    className={cn(
                      "flex shrink-0 items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 lg:w-full",
                      active
                        ? "bg-primary/10 font-medium text-primary"
                        : "text-muted-foreground hover:bg-accent hover:text-foreground",
                    )}
                  >
                    <item.icon className="h-4 w-4 shrink-0" />
                    <span className="min-w-0">
                      <span className="block whitespace-nowrap">{item.label}</span>
                      <span className="hidden text-xs font-normal text-muted-foreground lg:block">
                        {item.hint}
                      </span>
                    </span>
                    <ArrowRight
                      className={cn("ml-auto hidden h-3.5 w-3.5 lg:block", active ? "opacity-100" : "opacity-0")}
                      aria-hidden="true"
                    />
                  </button>
                )
              })}
            </div>
          </nav>

          <div className="min-w-0 space-y-6">
            {section === "auth" && (
              <>
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <KeyRound className="h-4 w-4 text-primary" />
                      <CardTitle>当前会话 Key</CardTitle>
                    </div>
                    <CardDescription>
                      将已有的 API Key 粘贴到此处，控制台将使用它进行所有的管理操作。
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Notice tone="info">
                      此 Key 只保存在当前浏览器的 localStorage（键名 <span className="font-mono">qwen2api_key</span>），
                      不会上传到服务器，也不会同步到其他设备。
                    </Notice>
                    <Field label="会话 Key">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                        <Input
                          type="password"
                          value={sessionKey}
                          onChange={e => setSessionKey(e.target.value)}
                          placeholder="sk-qwen-... 或默认管理员密钥 admin"
                          className="sm:flex-1"
                          autoComplete="off"
                        />
                        <div className="flex gap-2">
                          <Button onClick={handleSaveSessionKey} className="flex-1 sm:flex-none">保存到本地</Button>
                          <Button variant="ghost" onClick={handleClearSessionKey} className="flex-1 sm:flex-none">清除</Button>
                        </div>
                      </div>
                    </Field>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <ServerCrash className="h-4 w-4 text-primary" />
                      <CardTitle>连接信息</CardTitle>
                    </div>
                    <CardDescription>下游客户端应使用的网关地址。</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Field label="API 基础地址 (Base URL)">
                      <Input readOnly value={baseUrl} className="bg-muted font-mono text-muted-foreground" />
                    </Field>
                  </CardContent>
                </Card>
              </>
            )}

            {section === "runtime" && (
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal className="h-4 w-4 text-primary" />
                    <CardTitle>核心并发参数</CardTitle>
                  </div>
                  <CardDescription>
                    运行时并发槽位与排队阈值（部分项需在 config.json 中修改后重启生效）。
                  </CardDescription>
                </CardHeader>
                <CardContent className="divide-y divide-border py-0">
                  <div className="flex flex-col gap-2 py-4 sm:flex-row sm:items-center sm:justify-between">
                    <span className="text-sm font-medium">当前系统版本</span>
                    <span className="tabular font-mono text-sm text-muted-foreground">{settings?.version || "…"}</span>
                  </div>
                  <div className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="min-w-0 space-y-1">
                      <span className="text-sm font-medium">单账号最大并发 (max_inflight)</span>
                      <p className="text-xs text-muted-foreground">控制每个上游账号同时处理的请求数量，避免被封禁。</p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      <Input
                        type="number"
                        min="1"
                        max="10"
                        value={maxInflight}
                        onChange={e => setMaxInflight(Number(e.target.value))}
                        className="tabular h-9 w-20 text-center"
                        aria-label="单账号最大并发"
                      />
                      <Button size="sm" onClick={handleSaveConcurrency}>保存到服务器</Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {section === "routing" && (
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4 text-primary" />
                    <CardTitle>自动模型映射规则 (Model Aliases)</CardTitle>
                  </div>
                  <CardDescription>
                    下游传入的模型名称将被网关自动路由至以下千问实际模型。请使用标准 JSON 格式编辑。
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    rows={12}
                    value={modelAliases}
                    onChange={e => setModelAliases(e.target.value)}
                    className="code-surface min-h-[180px] font-mono"
                    spellCheck={false}
                    aria-label="模型映射 JSON"
                  />
                  <div className="flex justify-end">
                    <Button onClick={handleSaveAliases}>保存到服务器</Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {section === "developer" && (
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Code className="h-4 w-4 text-primary" />
                    <CardTitle>使用示例</CardTitle>
                  </div>
                  <CardDescription>各协议下调用本网关的最小可用请求示例。</CardDescription>
                </CardHeader>
                <CardContent className="min-w-0">
                  <div className="code-surface max-w-full min-w-0 overflow-x-auto whitespace-pre rounded-lg p-4 text-sm font-mono">
                    {curlExample}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </PageShell>
  )
}
