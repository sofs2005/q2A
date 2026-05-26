import { useEffect, useRef, useState } from "react"
import { Button } from "../components/ui/button"
import { Send, RefreshCw, Bot } from "lucide-react"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
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
    <div className="w-full space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">接口测试</h2>
          <p className="text-muted-foreground">在此测试您的 API 分发是否正常工作。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 text-sm bg-card border px-3 py-1.5 rounded-md">
              <span className="font-medium text-muted-foreground">模型:</span>
              <select
                value={model}
                onChange={e => setModel(e.target.value)}
                className="bg-transparent font-mono outline-none"
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
            {modelsError && <p className="text-xs text-red-500">{modelsError}</p>}
          </div>
          <div
            className="flex items-center gap-2 text-sm bg-card border px-3 py-1.5 rounded-md cursor-pointer"
            onClick={() => setStream(!stream)}
          >
            <input type="checkbox" checked={stream} onChange={() => {}} className="cursor-pointer" />
            <span className="font-medium">流式传输 (Stream)</span>
          </div>
          <Button variant="outline" onClick={() => setMessages([])}>
            <RefreshCw className="mr-2 h-4 w-4" /> 清空对话
          </Button>
        </div>
      </div>

      <div className="flex h-[calc(100vh-10rem)] flex-col overflow-hidden rounded-xl border bg-card shadow-sm">
        <div className="flex-1 overflow-y-auto p-6 space-y-6 flex flex-col">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground space-y-4">
              <Bot className="h-12 w-12 text-muted-foreground/30" />
              <p className="text-sm">
                {modelsError
                  ? "当前没有可用模型，请先检查 /v1/models 返回值。"
                  : "发送一条消息以开始测试，系统将通过 /v1/chat/completions 进行调用。"}
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[80%] rounded-xl px-4 py-3 text-sm shadow-sm
                ${msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : msg.error
                    ? "bg-red-500/10 border border-red-500/30 text-red-400"
                    : "bg-muted/30 border text-foreground"}`}>
                {msg.role === "assistant" && !msg.content && loading ? (
                  <span className="animate-pulse flex items-center gap-2 text-muted-foreground">
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

        <div className="p-4 border-t bg-muted/30 flex gap-3 items-center">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSend()}
            className="flex h-12 w-full rounded-md border border-input bg-background px-4 py-2 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            placeholder="输入测试消息..."
            disabled={loading}
          />
          <Button onClick={handleSend} disabled={loading || !input.trim() || !model || modelsLoading} className="h-12 px-6">
            {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </div>
  )
}
