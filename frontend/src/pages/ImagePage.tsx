import { useState } from "react"
import { Image as ImageIcon, RefreshCw, Download, Wand2, ExternalLink } from "lucide-react"
import { Button } from "../components/ui/button"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE, errorMessage, responseDetail, responseItems, type ImageResponseItem } from "../lib/api"
import { Card, CardContent } from "../components/ui/card"
import { Notice, SegmentedGroup, Textarea } from "../components/ui/field"
import { StatusBadge } from "../components/ui/badge"
import { PageShell } from "../components/ui/page-shell"

const ASPECT_RATIOS = [
  // auto 与官网默认一致：由上游按提示词自行决定比例，w/h 仅作占位
  { label: "自动", value: "auto", w: 0,    h: 0    },
  { label: "1:1",  value: "1:1",   w: 1024, h: 1024 },
  { label: "16:9", value: "16:9",  w: 1024, h: 576  },
  { label: "9:16", value: "9:16",  w: 576,  h: 1024 },
  { label: "4:3",  value: "4:3",   w: 1024, h: 768  },
  { label: "3:4",  value: "3:4",   w: 768,  h: 1024 },
]

interface GeneratedImage {
  url: string
  revised_prompt: string
  ratio: string
}

export default function ImagePage() {
  const [prompt, setPrompt] = useState("")
  const [ratio, setRatio] = useState("auto")
  const [n, setN] = useState(1)
  const [loading, setLoading] = useState(false)
  const [images, setImages] = useState<GeneratedImage[]>([])
  const [error, setError] = useState<string | null>(null)

  const selectedRatio = ASPECT_RATIOS.find(r => r.value === ratio)!
  // "auto" 直接透传给上游（与官网默认一致）；其余转成 WxH，由后端映射回宽高比
  const sizeStr = ratio === "auto" ? "auto" : `${selectedRatio.w}x${selectedRatio.h}`
  const sizeLabel = ratio === "auto" ? "自动（由上游决定）" : sizeStr

  const handleGenerate = async () => {
    if (!prompt.trim() || loading) return
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/v1/images/generations`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
        body: JSON.stringify({
          // 模型由后端 IMAGE_GENERATION_MODEL 决定，前端不写死
          prompt: prompt.trim(),
          n,
          size: sizeStr,
          response_format: "url",
        }),
      })

      const data: unknown = await res.json()
      if (!res.ok) {
        const detail = responseDetail(data, `HTTP ${res.status}`)
        setError(detail)
        toast.error(`生成失败: ${detail.slice(0, 80)}`)
        return
      }

      const newImages: GeneratedImage[] = responseItems<ImageResponseItem>(data).map(item => ({
        url: item.url,
        revised_prompt: item.revised_prompt || prompt,
        ratio,
      }))

      if (newImages.length === 0) {
        setError("未返回图片，请重试")
        toast.error("未返回图片，请重试")
        return
      }

      setImages(prev => [...newImages, ...prev])
      toast.success(`成功生成 ${newImages.length} 张图片`)
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
    a.download = `qwen_image_${Date.now()}_${idx}.png`
    a.target = "_blank"
    a.rel = "noopener noreferrer"
    a.click()
  }

  return (
    <PageShell
      actions={
        images.length > 0 ? (
          <Button variant="outline" onClick={() => setImages([])} disabled={loading}>
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
              <label htmlFor="image-prompt" className="text-sm font-medium">图片描述 (Prompt)</label>
              <Textarea
                id="image-prompt"
                rows={4}
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="描述你想生成的图片，例如：赛博朋克风格的猫咪，霓虹灯背景，超写实风格"
                className="resize-none"
                disabled={loading}
                onKeyDown={e => {
                  if (e.key === "Enter" && e.ctrlKey) handleGenerate()
                }}
              />
              <p className="text-xs text-muted-foreground">Ctrl+Enter 快速生成</p>
            </div>

            <SegmentedGroup
              label="图片比例"
              value={ratio}
              onChange={setRatio}
              disabled={loading}
              options={ASPECT_RATIOS.map(r => ({ label: r.label, value: r.value }))}
            />
            <SegmentedGroup
              label="生成数量"
              value={String(n)}
              onChange={value => setN(Number(value))}
              disabled={loading}
              options={[1, 2, 4].map(v => ({ label: `${v} 张`, value: String(v) }))}
            />

            <div className="flex items-center justify-between gap-3 border-t border-border pt-4">
              <span className="rounded-md border bg-muted/50 px-2 py-1 font-mono text-xs text-muted-foreground">
                {sizeLabel}
              </span>
              <Button
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                className="h-10 gap-2 px-6"
              >
                {loading
                  ? <><RefreshCw className="h-4 w-4 animate-spin" /> 生成中...</>
                  : <><Wand2 className="h-4 w-4" /> 生成图片</>
                }
              </Button>
            </div>

            {error && <Notice tone="error">{error}</Notice>}
          </CardContent>
        </Card>

        {/* 结果画布：空态 / 生成中 / 结果三态共用同一块区域 */}
        <div className="min-w-0 space-y-4">
          {images.length > 0 && (
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-sm font-semibold text-foreground">
                生成结果 ({images.length} 张)
              </h3>
            </div>
          )}

          {loading && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center gap-4 py-16 text-muted-foreground">
                <div className="relative">
                  <ImageIcon className="h-14 w-14 text-muted-foreground/20" />
                  <RefreshCw className="absolute -bottom-1 -right-1 h-6 w-6 animate-spin text-primary" />
                </div>
                <div className="text-center">
                  <p className="font-medium text-foreground">正在生成图片…</p>
                  <p className="mt-1 text-sm text-muted-foreground">图片生成通常需要 10-30 秒，请耐心等待</p>
                </div>
              </CardContent>
            </Card>
          )}

          {images.length > 0 && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {images.map((img, idx) => (
                <Card key={`${img.url}-${idx}`} className="overflow-hidden">
                  <div className="relative bg-muted/30">
                    <img
                      src={img.url}
                      alt={img.revised_prompt}
                      className="h-auto w-full object-contain"
                      loading="lazy"
                      onError={e => {
                        const target = e.currentTarget
                        target.style.display = "none"
                        target.nextElementSibling?.classList.remove("hidden")
                      }}
                    />
                    <div className="hidden items-center justify-center p-8 text-sm text-muted-foreground">
                      <ImageIcon className="mr-2 h-8 w-8" /> 图片加载失败
                    </div>
                  </div>
                  <CardContent className="space-y-3 py-3">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <StatusBadge tone="neutral" className="font-mono">{img.ratio}</StatusBadge>
                      <span className="truncate">{img.revised_prompt.slice(0, 80)}</span>
                    </div>
                    {/* 操作常驻可见，触屏与键盘用户同样可达 */}
                    <div className="flex flex-wrap items-center gap-2">
                      <Button size="sm" variant="outline" onClick={() => handleDownload(img.url, idx)} className="gap-1.5">
                        <Download className="h-3.5 w-3.5" /> 下载
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => window.open(img.url, "_blank")} className="gap-1.5">
                        <ExternalLink className="h-3.5 w-3.5" /> 新窗口打开
                      </Button>
                    </div>
                    <div className="truncate font-mono text-xs text-muted-foreground/80">{img.url}</div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {images.length === 0 && !loading && (
            <Card>
              <CardContent className="flex flex-col items-center gap-4 py-20 text-muted-foreground">
                <ImageIcon className="h-14 w-14 text-muted-foreground/20" aria-hidden="true" />
                <div className="text-center">
                  <p className="font-medium text-foreground">还没有生成图片</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    在左侧填写描述与比例，点击「生成图片」开始创作。
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
