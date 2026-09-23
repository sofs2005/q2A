import { useState } from "react"
import { Film, RefreshCw, Download, Wand2, ExternalLink } from "lucide-react"
import { Button } from "../components/ui/button"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE, errorMessage, responseDetail, responseItems, sseErrorDetail, type VideoResponseItem } from "../lib/api"
import { Card, CardContent } from "../components/ui/card"
import { Notice, SegmentedGroup, Textarea } from "../components/ui/field"
import { StatusBadge } from "../components/ui/badge"
import { PageShell } from "../components/ui/page-shell"

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
async function readSSEResult(res: Response): Promise<unknown> {
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
        const errData: unknown = await res.json().catch(() => ({}))
        const detail = responseDetail(errData, `HTTP ${res.status}`)
        setError(detail)
        toast.error(`生成失败: ${detail.slice(0, 80)}`)
        return
      }

      // SSE 心跳流：跳过心跳注释，解析最终 data 事件
      const data = await readSSEResult(res)
      const sseError = sseErrorDetail(data)
      if (sseError) {
        setError(sseError)
        toast.error(`生成失败: ${sseError.slice(0, 80)}`)
        return
      }

      const newVideos: GeneratedVideo[] = responseItems<VideoResponseItem>(data).map(item => ({
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
    } catch (err: unknown) {
      const msg = errorMessage(err)
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
    <PageShell
      actions={
        videos.length > 0 ? (
          <Button variant="outline" onClick={() => setVideos([])} disabled={loading}>
            清空结果
          </Button>
        ) : null
      }
    >
      {/* 桌面：左侧控制面板 + 右侧结果画布；窄屏按“先参数后结果”堆叠 */}
      <div className="grid gap-5 lg:grid-cols-[22rem_minmax(0,1fr)] lg:items-start">
        <Card className="lg:sticky lg:top-[5.5rem]">
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="video-prompt" className="text-sm font-medium">视频描述 (Prompt)</label>
              <Textarea
                id="video-prompt"
                rows={4}
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

            <div className="border-t border-border pt-4">
              <Button
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                className="h-10 w-full gap-2"
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

        {/* 结果画布：空态 / 生成中 / 结果三态共用同一块区域 */}
        <div className="min-w-0 space-y-4">
          {videos.length > 0 && (
            <h3 className="text-sm font-semibold text-foreground">生成结果 ({videos.length} 个)</h3>
          )}

          {loading && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center gap-4 py-16 text-muted-foreground">
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

          {videos.length > 0 && (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
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
          )}

          {videos.length === 0 && !loading && (
            <Card>
              <CardContent className="flex flex-col items-center gap-4 py-20 text-muted-foreground">
                <Film className="h-14 w-14 text-muted-foreground/20" aria-hidden="true" />
                <div className="text-center">
                  <p className="font-medium text-foreground">还没有生成视频</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    在左侧填写描述、比例与时长，点击「生成视频」开始创作。
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </PageShell>
  )
}
