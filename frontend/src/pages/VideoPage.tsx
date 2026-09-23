import { useState } from "react"
import { Film, RefreshCw, Download, Wand2, ExternalLink } from "lucide-react"
import { Button } from "../components/ui/button"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { Card, CardContent } from "../components/ui/card"
import { Notice, SegmentedGroup, Textarea } from "../components/ui/field"
import { StatusBadge } from "../components/ui/badge"

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
        <h2 className="text-2xl font-semibold tracking-tight">视频生成</h2>
        <p className="mt-1 text-sm text-muted-foreground">通过 Qwen3.6-Plus 生成 AI 视频，支持多种比例与时长。</p>
      </div>

      {/* 输入区域 */}
      <Card>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="video-prompt" className="text-sm font-medium">视频描述 (Prompt)</label>
            <Textarea
              id="video-prompt"
              rows={3}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="描述你想生成的视频，例如：一只白色小猫在樱花树下奔跑，阳光洒落，电影感运镜"
              className="resize-none"
              disabled={loading}
              onKeyDown={e => {
                if (e.key === "Enter" && e.ctrlKey) handleGenerate()
              }}
            />
            <p className="text-xs text-muted-foreground">Ctrl+Enter 快速生成</p>
          </div>

          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="flex flex-wrap gap-5">
              <SegmentedGroup
                label="视频比例"
                value={ratio}
                onChange={setRatio}
                disabled={loading}
                options={ASPECT_RATIOS}
              />
              <SegmentedGroup
                label="时长（秒）"
                value={String(duration)}
                onChange={value => setDuration(Number(value))}
                disabled={loading}
                options={DURATIONS.map(v => ({ label: `${v}s`, value: String(v) }))}
              />
              <SegmentedGroup
                label="生成数量"
                value={String(n)}
                onChange={value => setN(Number(value))}
                disabled={loading}
                options={[1, 2].map(v => ({ label: `${v} 个`, value: String(v) }))}
              />
            </div>

            <Button
              onClick={handleGenerate}
              disabled={loading || !prompt.trim()}
              className="h-10 shrink-0 gap-2 px-6"
            >
              {loading
                ? <><RefreshCw className="h-4 w-4 animate-spin" /> 生成中...</>
                : <><Wand2 className="h-4 w-4" /> 生成视频</>
              }
            </Button>
          </div>

          {error && <Notice tone="error">{error}</Notice>}
        </CardContent>
      </Card>

      {/* 加载状态占位 */}
      {loading && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center gap-4 py-12 text-muted-foreground">
            <div className="relative">
              <Film className="h-14 w-14 text-muted-foreground/20" />
              <RefreshCw className="absolute -bottom-1 -right-1 h-6 w-6 animate-spin text-primary" />
            </div>
            <div className="text-center">
              <p className="font-medium text-foreground">正在生成视频…</p>
              <p className="mt-1 text-sm text-muted-foreground">视频生成耗时较长（最长约 7 分钟），请保持页面打开耐心等待</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 视频展示区 */}
      {videos.length > 0 && !loading && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">生成结果 ({videos.length} 个)</h3>
            <Button variant="ghost" size="sm" onClick={() => setVideos([])}>清空</Button>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {videos.map((vid, idx) => (
              <Card key={`${vid.url}-${idx}`} className="overflow-hidden">
                <div className="relative bg-black">
                  <video src={vid.url} controls playsInline className="h-auto w-full" />
                </div>
                <CardContent className="space-y-3 py-3">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <StatusBadge tone="neutral" className="font-mono">{vid.ratio}</StatusBadge>
                    <StatusBadge tone="neutral" className="font-mono">{vid.duration}s</StatusBadge>
                    <span className="min-w-0 truncate">{vid.revised_prompt.slice(0, 80)}</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button size="sm" variant="outline" onClick={() => handleDownload(vid.url, idx)} className="gap-1.5">
                      <Download className="h-3.5 w-3.5" /> 下载
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => window.open(vid.url, "_blank")} className="gap-1.5">
                      <ExternalLink className="h-3.5 w-3.5" /> 新窗口打开
                    </Button>
                  </div>
                  <div className="truncate font-mono text-xs text-muted-foreground/80">{vid.url}</div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* 空状态 */}
      {videos.length === 0 && !loading && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 py-14 text-muted-foreground">
            <Film className="h-14 w-14 text-muted-foreground/20" />
            <div className="text-center">
              <p className="font-medium text-foreground">还没有生成视频</p>
              <p className="mt-1 text-sm text-muted-foreground">在上方输入描述，点击「生成视频」开始创作</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
