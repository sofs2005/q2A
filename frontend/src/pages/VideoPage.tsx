import { useState } from "react"
import { Film, RefreshCw, Download, Wand2 } from "lucide-react"
import { Button } from "../components/ui/button"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"

const ASPECT_RATIOS = [
  { label: "16:9", value: "16:9" },
  { label: "9:16", value: "9:16" },
  { label: "1:1",  value: "1:1"  },
  { label: "4:3",  value: "4:3"  },
  { label: "3:4",  value: "3:4"  },
]

const DURATIONS = [3, 5, 10]

interface GeneratedVideo {
  url: string
  revised_prompt: string
  ratio: string
  duration: number
}

// 读取 SSE 心跳流：忽略 `: heartbeat` 注释，返回首个 data 事件的 JSON
async function readSSEResult(res: Response): Promise<any> {
  const reader = res.body?.getReader()
  if (!reader) return null
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      for (const line of block.split("\n")) {
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim()
          if (payload) {
            try { return JSON.parse(payload) } catch { return null }
          }
        }
        // 以 ':' 开头的是心跳注释，忽略
      }
    }
  }
  return null
}

export default function VideoPage() {
  const [prompt, setPrompt] = useState("")
  const [ratio, setRatio] = useState("16:9")
  const [duration, setDuration] = useState(5)
  const [n, setN] = useState(1)
  const [loading, setLoading] = useState(false)
  const [videos, setVideos] = useState<GeneratedVideo[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleGenerate = async () => {
    if (!prompt.trim() || loading) return
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/v1/videos/generations`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
        body: JSON.stringify({
          model: "qwen3.6-plus",
          prompt: prompt.trim(),
          n,
          ratio,
          duration,
        }),
      })

      // 鉴权/参数错误以普通 JSON 返回（非 SSE）
      if (!res.ok && !res.headers.get("content-type")?.includes("text/event-stream")) {
        const errData = await res.json().catch(() => ({}))
        const detail = errData?.detail || errData?.error || `HTTP ${res.status}`
        setError(String(detail))
        toast.error(`生成失败: ${String(detail).slice(0, 80)}`)
        return
      }

      // SSE 心跳流：跳过心跳注释，解析最终 data 事件
      const data = await readSSEResult(res)
      if (data?.error) {
        const detail = data.error?.message || data.error || "生成失败"
        setError(String(detail))
        toast.error(`生成失败: ${String(detail).slice(0, 80)}`)
        return
      }

      const newVideos: GeneratedVideo[] = (data?.data || []).map((item: any) => ({
        url: item.url,
        revised_prompt: item.revised_prompt || prompt,
        ratio: item.ratio || ratio,
        duration: item.duration || duration,
      }))

      if (newVideos.length === 0) {
        setError("未返回视频，请重试")
        toast.error("未返回视频，请重试")
        return
      }

      setVideos(prev => [...newVideos, ...prev])
      toast.success(`成功生成 ${newVideos.length} 个视频`)
    } catch (err: any) {
      const msg = err.message || "网络错误"
      setError(msg)
      toast.error(`生成失败: ${msg}`)
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = (url: string, idx: number) => {
    const a = document.createElement("a")
    a.href = url
    a.download = `qwen_video_${Date.now()}_${idx}.mp4`
    a.target = "_blank"
    a.rel = "noopener noreferrer"
    a.click()
  }

  return (
    <div className="w-full space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">视频生成</h2>
        <p className="text-muted-foreground">通过 Qwen3.6-Plus 生成 AI 视频，支持多种比例与时长。</p>
      </div>

      {/* 输入区域 */}
      <div className="rounded-xl border bg-card shadow-sm p-6 space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">视频描述 (Prompt)</label>
          <textarea
            rows={3}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="描述你想生成的视频，例如：一只白色小猫在樱花树下奔跑，阳光洒落，电影感运镜"
            className="flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm resize-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            disabled={loading}
            onKeyDown={e => {
              if (e.key === "Enter" && e.ctrlKey) handleGenerate()
            }}
          />
          <p className="text-xs text-muted-foreground">Ctrl+Enter 快速生成</p>
        </div>

        <div className="flex flex-wrap gap-4 items-end">
          {/* 比例选择 */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium">视频比例</label>
            <div className="flex gap-2">
              {ASPECT_RATIOS.map(r => (
                <button
                  key={r.value}
                  onClick={() => setRatio(r.value)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium border transition-all ${
                    ratio === r.value
                      ? "bg-primary text-primary-foreground border-primary shadow-sm"
                      : "bg-background border-border text-muted-foreground hover:text-foreground hover:border-foreground/30"
                  }`}
                  disabled={loading}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>

          {/* 时长选择 */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium">时长（秒）</label>
            <div className="flex gap-2">
              {DURATIONS.map(v => (
                <button
                  key={v}
                  onClick={() => setDuration(v)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium border transition-all ${
                    duration === v
                      ? "bg-primary text-primary-foreground border-primary shadow-sm"
                      : "bg-background border-border text-muted-foreground hover:text-foreground hover:border-foreground/30"
                  }`}
                  disabled={loading}
                >
                  {v}s
                </button>
              ))}
            </div>
          </div>

          {/* 数量选择 */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium">生成数量</label>
            <div className="flex gap-2">
              {[1, 2].map(v => (
                <button
                  key={v}
                  onClick={() => setN(v)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium border transition-all ${
                    n === v
                      ? "bg-primary text-primary-foreground border-primary shadow-sm"
                      : "bg-background border-border text-muted-foreground hover:text-foreground hover:border-foreground/30"
                  }`}
                  disabled={loading}
                >
                  {v} 个
                </button>
              ))}
            </div>
          </div>

          {/* 生成按钮 */}
          <Button
            onClick={handleGenerate}
            disabled={loading || !prompt.trim()}
            className="ml-auto h-10 px-6 gap-2"
          >
            {loading
              ? <><RefreshCw className="h-4 w-4 animate-spin" /> 生成中...</>
              : <><Wand2 className="h-4 w-4" /> 生成视频</>
            }
          </Button>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="rounded-md bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 text-sm">
            {error}
          </div>
        )}
      </div>

      {/* 加载状态占位 */}
      {loading && (
        <div className="rounded-xl border bg-card shadow-sm p-8">
          <div className="flex flex-col items-center justify-center gap-4 text-muted-foreground">
            <div className="relative">
              <Film className="h-16 w-16 text-muted-foreground/20" />
              <RefreshCw className="h-6 w-6 animate-spin absolute -bottom-1 -right-1 text-primary" />
            </div>
            <div className="text-center">
              <p className="font-medium">正在生成视频...</p>
              <p className="text-sm text-muted-foreground/70 mt-1">视频生成耗时较长（最长约 7 分钟），请保持页面打开耐心等待</p>
            </div>
          </div>
        </div>
      )}

      {/* 视频展示区 */}
      {videos.length > 0 && !loading && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">生成结果 ({videos.length} 个)</h3>
            <Button variant="ghost" size="sm" onClick={() => setVideos([])}>
              清空
            </Button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {videos.map((vid, idx) => (
              <div key={`${vid.url}-${idx}`} className="rounded-xl border bg-card shadow-sm overflow-hidden">
                <div className="relative bg-black">
                  <video
                    src={vid.url}
                    controls
                    playsInline
                    className="w-full h-auto"
                  />
                </div>
                <div className="p-3 space-y-2">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="bg-muted rounded px-1.5 py-0.5 font-mono">{vid.ratio}</span>
                    <span className="bg-muted rounded px-1.5 py-0.5 font-mono">{vid.duration}s</span>
                    <span className="truncate">{vid.revised_prompt.slice(0, 80)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => handleDownload(vid.url, idx)}
                      className="gap-1.5"
                    >
                      <Download className="h-3.5 w-3.5" /> 下载
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => window.open(vid.url, "_blank")}
                    >
                      在新窗口打开
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground font-mono truncate">{vid.url}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 空状态 */}
      {videos.length === 0 && !loading && (
        <div className="rounded-xl border bg-card/50 shadow-sm p-12">
          <div className="flex flex-col items-center gap-4 text-muted-foreground">
            <Film className="h-16 w-16 text-muted-foreground/20" />
            <div className="text-center">
              <p className="font-medium">还没有生成视频</p>
              <p className="text-sm text-muted-foreground/70 mt-1">在上方输入描述，点击「生成视频」开始创作</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
