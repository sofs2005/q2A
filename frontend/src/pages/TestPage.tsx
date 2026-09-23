import { useEffect, useRef, useState, type JSX } from "react"
import { Button } from "../components/ui/button"
import { Send, RefreshCw, Bot } from "lucide-react"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { PageShell } from "../components/ui/page-shell"
import { toast } from "sonner"

// 渲染消息内容：自动把 Markdown 图片和图片 URL 渲染成 <img>
function MessageContent({ content }: { content: string }) {
  type Seg = { start: number; end: number; url: string }
  const segs: Seg[] = []
  const fullRe = /!\[[^\]]*\]\((https?:\/\/[^)\s]+)\)|(https?:\/\/[^\s"<>]+\.(?:jpg|jpeg|png|webp|gif)[^\s"<>]*)/gi
  let m: RegExpExecArray | null
  while ((m = fullRe.exec(content)) !== null) {
    segs.push({ start: m.index, end: m.index + m[0].length, url: (m[1] || m[2]) as string })
  }

  if (segs.length === 0) {
    return <div className="whitespace-pre-wrap leading-relaxed">{content}</div>
  }

  const nodes: JSX.Element[] = []
  let cursor = 0
  segs.forEach((seg, i) => {
    if (seg.start > cursor) {
      nodes.push(<span key={"t" + i}>{content.slice(cursor, seg.start)}</span>)
    }
    nodes.push(
      <div key={"i" + i} className="my-2">
        <img
          src={seg.url}
          alt="generated"
          className="max-w-full rounded-lg shadow-md border"
          loading="lazy"
          onError={e => { (e.currentTarget as HTMLImageElement).style.display = "none" }}
        />
        <div className="text-xs text-muted-foreground mt-1 break-all font-mono">{seg.url}</div>
      </div>
    )
    cursor = seg.end
  })
  if (cursor < content.length) {
    nodes.push(<span key="tail">{content.slice(cursor)}</span>)
  }
  return <div className="whitespace-pre-wrap leading-relaxed">{nodes}</div>
}

export default function TestPage() {
  const [messages, setMessages] = useState<{ role: string; content: string; error?: boolean }[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState("")
  const [modelsLoading, setModelsLoading] = useState(true)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [stream, setStream] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    let alive = true

    const loadModels = async () => {
      setModelsLoading(true)
      setModelsError(null)

      try {
        const res = await fetch(`${API_BASE}/v1/models`, {
          headers: getAuthHeader(),
        })
        const text = await res.text()
        let data: unknown = null

        if (text) {
          try {
            data = JSON.parse(text)
          } catch {
            data = text
          }
        }

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }

        const nextModels = Array.isArray((data as { data?: unknown } | null)?.data)
          ? ((data as { data?: Array<{ id?: unknown }> }).data || [])
              .map(item => (typeof item?.id === "string" ? item.id : ""))
              .filter((id): id is string => id.length > 0)
          : []

        if (!alive) return

        setModels(nextModels)
        setModel(prev => {
          if (prev && nextModels.includes(prev)) return prev
          return nextModels[0] ?? ""
        })

        if (nextModels.length === 0) {
          setModelsError("暂无可用模型")
        }
      } catch {
        if (!alive) return
        setModels([])
        setModel("")
        setModelsError("模型列表加载失败")
      } finally {
        if (alive) setModelsLoading(false)
      }
    }

    loadModels()
    return () => {
      alive = false
    }
  }, [])

  const handleSend = async () => {
    if (!input.trim() || loading || modelsLoading || !model) return
    const userMsg = { role: "user", content: input }
    setMessages(prev => [...prev, userMsg])
    setInput("")
    setLoading(true)

    try {
      if (!stream) {
        const res = await fetch(`${API_BASE}/v1/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...getAuthHeader() },
          body: JSON.stringify({ model, messages: [...messages, userMsg], stream: false })
        })
        const data = await res.json()
        if (data.error) {
          setMessages(prev => [...prev, { role: "assistant", content: `❌ ${data.error}`, error: true }])
        } else if (data.choices?.[0]) {
          setMessages(prev => [...prev, data.choices[0].message])
        } else {
          setMessages(prev => [...prev, { role: "assistant", content: `❌ 未知响应: ${JSON.stringify(data)}`, error: true }])
        }
      } else {
        const res = await fetch(`${API_BASE}/v1/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...getAuthHeader() },
          body: JSON.stringify({ model, messages: [...messages, userMsg], stream: true })
        })

        if (!res.ok) {
          const errText = await res.text()
          setMessages(prev => [...prev, { role: "assistant", content: `❌ HTTP ${res.status}: ${errText}`, error: true }])
          return
        }

        if (!res.body) throw new Error("No response body")

        setMessages(prev => [...prev, { role: "assistant", content: "" }])
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let hasContent = false

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          for (const rawLine of chunk.split("\n")) {
            const line = rawLine.trim()
            if (!line || line.startsWith(":") || line === "data: [DONE]") continue
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6))
                if (data.error) {
                  setMessages(prev => {
                    const msgs = [...prev]
                    msgs[msgs.length - 1] = { role: "assistant", content: `❌ ${data.error}`, error: true }
                    return msgs
                  })
                  hasContent = true
                  break
                }
                const content: string = data.choices?.[0]?.delta?.content ?? ""
                if (content) {
                  hasContent = true
                  setMessages(prev => {
                    const msgs = [...prev]
                    const last = msgs[msgs.length - 1]
                    msgs[msgs.length - 1] = { ...last, content: last.content + content }
                    return msgs
                  })
                }
              } catch {
                /* skip */
              }
            }
          }
        }

        if (!hasContent) {
          setMessages(prev => {
            const msgs = [...prev]
            msgs[msgs.length - 1] = { role: "assistant", content: "❌ 响应为空（账号可能未激活或无可用账号）", error: true }
            return msgs
          })
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "未知网络错误"
      toast.error(`网络错误: ${message}`)
      setMessages(prev => [...prev, { role: "assistant", content: `❌ 网络错误: ${message}`, error: true }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageShell
      actions={
        <>
          {/* 会话配置：模型与传输方式，与消息舞台分离 */}
          <div className="flex h-10 items-center gap-2 rounded-md border bg-background px-3 text-sm">
            <label htmlFor="test-model-select" className="shrink-0 font-medium text-muted-foreground">模型</label>
            <select
              id="test-model-select"
              value={model}
              onChange={e => setModel(e.target.value)}
              className="min-w-0 max-w-[12rem] bg-transparent font-mono text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={modelsLoading || models.length === 0}
            >
              {modelsLoading ? (
                <option value="">加载模型中...</option>
              ) : models.length > 0 ? (
                models.map(item => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))
              ) : (
                <option value="">暂无可用模型</option>
              )}
            </select>
          </div>
          {/* 语义化开关：label 关联 checkbox，键盘可用 */}
          <label
            htmlFor="test-stream-toggle"
            className="flex h-10 cursor-pointer items-center gap-2 rounded-md border bg-background px-3 text-sm transition-colors hover:bg-accent focus-within:ring-2 focus-within:ring-ring/40"
          >
            <input
              id="test-stream-toggle"
              type="checkbox"
              checked={stream}
              onChange={e => setStream(e.target.checked)}
              className="h-4 w-4 cursor-pointer rounded border-input accent-[hsl(var(--primary))]"
            />
            <span className="font-medium">流式传输</span>
          </label>
          <Button variant="outline" onClick={() => setMessages([])} disabled={messages.length === 0}>
            <RefreshCw className="mr-2 h-4 w-4" /> 清空对话
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        {modelsError && <p className="text-xs text-red-600 dark:text-red-400">{modelsError}</p>}

        {/* 全高对话舞台：消息区占主体，composer 固定在底部 */}
        <div className="flex h-[calc(100vh-13rem)] min-h-[26rem] flex-col overflow-hidden rounded-xl border border-border bg-card">
          <div className="flex flex-1 flex-col space-y-6 overflow-y-auto p-4 sm:p-6">
            {messages.length === 0 && (
              <div className="flex h-full flex-col items-center justify-center space-y-4 text-center text-muted-foreground">
                <Bot className="h-12 w-12 text-muted-foreground/30" aria-hidden="true" />
                <div className="space-y-1">
                  <p className="text-sm font-medium text-foreground">开始一轮测试对话</p>
                  <p className="text-sm">
                    {modelsError
                      ? "当前没有可用模型，请先检查 /v1/models 返回值。"
                      : "发送一条消息，系统将通过 /v1/chat/completions 调用网关。"}
                  </p>
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[85%] rounded-xl px-4 py-3 text-sm sm:max-w-[80%]
                  ${msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : msg.error
                      ? "border border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300"
                      : "border bg-muted/40 text-foreground"}`}>
                  {msg.role === "assistant" && !msg.content && loading ? (
                    <span className="flex animate-pulse items-center gap-2 text-muted-foreground">
                      <Bot className="h-4 w-4" /> 思考中...
                    </span>
                  ) : msg.role === "assistant" && !msg.error ? (
                    <MessageContent content={msg.content} />
                  ) : (
                    <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          <div className="flex items-center gap-3 border-t border-border bg-muted/30 p-3 sm:p-4">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSend()}
              className="flex h-11 w-full min-w-0 rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground/70 focus-visible:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50 sm:px-4"
              placeholder="输入测试消息..."
              aria-label="测试消息"
              disabled={loading}
            />
            <Button onClick={handleSend} disabled={loading || !input.trim() || !model || modelsLoading} className="h-11 shrink-0 gap-2 px-4 sm:px-6">
              {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              <span className="hidden sm:inline">{loading ? "发送中" : "发送"}</span>
            </Button>
          </div>
        </div>
      </div>
    </PageShell>
  )
}
