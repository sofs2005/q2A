import * as React from "react"
import { createPortal } from "react-dom"
import { ChevronLeft, ChevronRight, Download, ExternalLink, X, ZoomIn, ZoomOut } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "./button"

/** 大图查看所需的最小媒体描述；页面把自己的数据结构映射成它即可复用。 */
export type LightboxMedia = {
  url: string
  kind: "image" | "video"
  caption?: string
  downloadName?: string
}

/**
 * 全屏媒体查看层。
 *
 * 由外部通过 `index` 控制开关：`null` 表示关闭，因此组件本身无内部开合状态，
 * 页面切换筛选或清空结果时只需把 index 置空即可，不会残留遮罩。
 */
export function Lightbox({
  media,
  index,
  onIndexChange,
  onClose,
}: {
  media: LightboxMedia[]
  index: number | null
  onIndexChange: (index: number) => void
  onClose: () => void
}) {
  // 关闭或越界时不挂载内部组件，避免为其跑一堆无意义的副作用
  if (index === null || index < 0 || index >= media.length) return null

  // 渲染到 body：页面内容区在 z-0 的层叠上下文里，留在原地的话侧栏会盖住遮罩
  return createPortal(
    <LightboxDialog media={media} index={index} onIndexChange={onIndexChange} onClose={onClose} />,
    document.body,
  )
}

function LightboxDialog({
  media,
  index,
  onIndexChange,
  onClose,
}: {
  media: LightboxMedia[]
  index: number
  onIndexChange: (index: number) => void
  onClose: () => void
}) {
  const closeRef = React.useRef<HTMLButtonElement>(null)
  const [zoomed, setZoomed] = React.useState(false)

  const total = media.length
  const canSwitch = total > 1

  // 切图时回到「适应屏幕」，否则上一张的放大状态会莫名其妙地跟到下一张
  React.useEffect(() => {
    setZoomed(false)
  }, [index])

  // 打开期间锁住背景滚动；还原成原值而不是清空，避免覆盖调用方已有的内联样式
  React.useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      document.body.style.overflow = previous
    }
  }, [])

  // 键盘可达：Escape 关闭、左右方向键切图（只有一张时切图键不生效）
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose()
        return
      }
      if (!canSwitch) return
      if (event.key === "ArrowLeft") onIndexChange((index - 1 + total) % total)
      if (event.key === "ArrowRight") onIndexChange((index + 1) % total)
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [canSwitch, index, total, onClose, onIndexChange])

  // 打开即把焦点交给关闭按钮：键盘用户不必先 Tab 过整条工具条
  React.useEffect(() => {
    closeRef.current?.focus()
  }, [])

  const current = media[index]
  // 列表可能在打开期间被删除变短，兜底关掉而不是渲染空白
  if (!current) return null

  const isVideo = current.kind === "video"

  // 与页面里 handleDownload 一致：走临时 <a download>，跨域资源另有新窗口兜底
  const handleDownload = () => {
    const a = document.createElement("a")
    a.href = current.url
    a.download =
      current.downloadName ||
      `qwen_${current.kind}_${Date.now()}${isVideo ? ".mp4" : ".png"}`
    a.target = "_blank"
    a.rel = "noopener noreferrer"
    a.click()
  }

  // 只有点在空白容器本身（而非图片/按钮/视频）才关闭
  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) onClose()
  }

  const actionClass = "gap-1.5 text-white/80 hover:text-white"

  return (
    <div
      className="fixed inset-0 z-[100] flex flex-col bg-black/90 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="媒体预览"
      onClick={handleBackdropClick}
    >
      {/* 顶部条：黑底上用白色幽灵按钮，避免 primary 配色压住画面本身 */}
      <div className="flex shrink-0 items-center justify-between gap-3 px-4 py-3">
        <span className="shrink-0 text-sm text-white/70">
          第 {index + 1} / {total} {isVideo ? "个" : "张"}
        </span>
        <div className="flex flex-wrap items-center justify-end gap-1">
          {isVideo ? null : (
            <Button
              variant="ghost"
              size="sm"
              className={actionClass}
              onClick={() => setZoomed(value => !value)}
              aria-label={zoomed ? "缩小到适应屏幕" : "按原始尺寸查看"}
            >
              {zoomed ? <ZoomOut className="h-4 w-4" /> : <ZoomIn className="h-4 w-4" />}
              {zoomed ? "适应屏幕" : "原始尺寸"}
            </Button>
          )}
          <Button variant="ghost" size="sm" className={actionClass} onClick={handleDownload}>
            <Download className="h-4 w-4" /> 下载
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className={actionClass}
            onClick={() => window.open(current.url, "_blank", "noopener,noreferrer")}
          >
            <ExternalLink className="h-4 w-4" /> 新窗口打开
          </Button>
          <Button
            ref={closeRef}
            variant="ghost"
            size="icon"
            className="text-white/80 hover:text-white"
            onClick={onClose}
            aria-label="关闭预览"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* 内容区：外层 relative 承载左右切换按钮，内层滚动容器承载放大后的图片 */}
      <div className="relative flex min-h-0 flex-1">
        <div
          className={cn(
            "flex min-h-0 flex-1 p-4",
            zoomed ? "overflow-auto" : "items-center justify-center overflow-hidden",
          )}
          // 放大后容器可滚动，此时不接管点击，否则点滚动条也会被当成「点空白」
          onClick={zoomed ? undefined : handleBackdropClick}
        >
          {isVideo ? (
            <video
              src={current.url}
              controls
              autoPlay
              playsInline
              className="max-h-[86vh] max-w-[94vw]"
            />
          ) : (
            <button
              type="button"
              onClick={event => {
                event.stopPropagation()
                setZoomed(value => !value)
              }}
              aria-label={zoomed ? "缩小到适应屏幕" : "按原始尺寸查看"}
              className={cn(
                "block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60",
                zoomed ? "m-auto cursor-zoom-out" : "cursor-zoom-in",
              )}
            >
              <img
                src={current.url}
                alt={current.caption || "预览图片"}
                className={cn(
                  "block object-contain",
                  zoomed ? "max-h-none max-w-none" : "max-h-[86vh] max-w-[94vw]",
                )}
              />
            </button>
          )}
        </div>

        {canSwitch && (
          <>
            <button
              type="button"
              onClick={() => onIndexChange((index - 1 + total) % total)}
              aria-label="上一张"
              className="absolute left-3 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-2 text-white/80 transition-colors hover:bg-black/70 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60"
            >
              <ChevronLeft className="h-6 w-6" />
            </button>
            <button
              type="button"
              onClick={() => onIndexChange((index + 1) % total)}
              aria-label="下一张"
              className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-2 text-white/80 transition-colors hover:bg-black/70 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60"
            >
              <ChevronRight className="h-6 w-6" />
            </button>
          </>
        )}
      </div>

      {current.caption ? (
        <p className="line-clamp-2 shrink-0 px-6 pb-4 text-center text-sm text-white/60">
          {current.caption}
        </p>
      ) : null}
    </div>
  )
}
