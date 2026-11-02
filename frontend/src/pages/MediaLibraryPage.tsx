import { useEffect, useRef, useState } from "react"
import { Link } from "react-router-dom"
import {
  CheckSquare,
  Image as ImageIcon,
  Maximize2,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  Square,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"
import { Button } from "../components/ui/button"
import { Card, CardContent } from "../components/ui/card"
import { Input, Notice, SegmentedGroup, Select } from "../components/ui/field"
import { StatusBadge } from "../components/ui/badge"
import { PageShell } from "../components/ui/page-shell"
import { Lightbox } from "../components/ui/lightbox"
import { getAuthHeader } from "../lib/auth"
import {
  API_BASE,
  errorMessage,
  mediaListFrom,
  responseDetail,
  type MediaItem,
  type MediaStats,
} from "../lib/api"
import { cn, formatBytes, formatTimestamp } from "../lib/utils"

type MediaKind = "all" | "image" | "video"
type SortMode = "newest" | "oldest" | "largest"

/** 单删与批删共用同一套确认流程，ids 为空表示未打开弹窗。 */
type DeleteTarget = { ids: string[]; label: string }

/** 当前筛选条件；全选/反选与列表请求共用同一份，避免两边口径漂移。 */
type MediaFilters = {
  kind: MediaKind
  query: string
  sinceDays: string
  minSize: string
  sort: SortMode
}

const PAGE_SIZE = 24
const SEARCH_DEBOUNCE_MS = 300
// 全选「未加载的部分」时按 100 条/页翻；上限 50 页（5000 条），防止极端情况把浏览器拖死
const SELECT_ALL_PAGE_SIZE = 100
const SELECT_ALL_MAX_PAGES = 50

const KIND_OPTIONS = [
  { label: "全部", value: "all" },
  { label: "图片", value: "image" },
  { label: "视频", value: "video" },
]

// 时间窗与体积下限都是「清理时的常见动机」：清今天的废图、清占地方的大家伙
const SINCE_OPTIONS = [
  { label: "全部时间", value: "0" },
  { label: "今天", value: "1" },
  { label: "近 7 天", value: "7" },
  { label: "近 30 天", value: "30" },
]

const MIN_SIZE_OPTIONS = [
  { label: "不限体积", value: "0" },
  { label: "大于 1 MB", value: String(1024 * 1024) },
  { label: "大于 5 MB", value: String(5 * 1024 * 1024) },
  { label: "大于 20 MB", value: String(20 * 1024 * 1024) },
]

const SORT_OPTIONS = [
  { label: "最新优先", value: "newest" },
  { label: "最早优先", value: "oldest" },
  { label: "体积最大优先", value: "largest" },
]

const EMPTY_STATS: MediaStats = { image: 0, video: 0, bytes: 0 }

/** SegmentedGroup 只回传 string，这里收敛回受限类型。 */
function toKind(value: string): MediaKind {
  return value === "image" || value === "video" ? value : "all"
}

/** Select 同样只回传 string。 */
function toSort(value: string): SortMode {
  return value === "oldest" || value === "largest" ? value : "newest"
}

/** 单删返回 { deleted: true }，批删返回 { deleted: n }：统一折算成条数用于提示与计数。 */
function deletedCount(payload: unknown, requested: number): number {
  if (typeof payload !== "object" || payload === null) return requested
  const raw = (payload as Record<string, unknown>).deleted
  if (typeof raw === "number" && Number.isFinite(raw)) return raw
  return raw === true ? 1 : requested
}

/**
 * 拼列表请求参数。抽成纯函数是为了让 useEffect 的依赖只包含基础类型，
 * 同时让「全选时翻页拉 id」和「列表加载」用完全相同的筛选条件。
 */
function buildMediaQuery(filters: MediaFilters, limit: number, offset: number): string {
  const params = new URLSearchParams({
    kind: filters.kind,
    limit: String(limit),
    offset: String(offset),
  })
  const query = filters.query.trim()
  if (query) params.set("q", query)
  if (filters.sinceDays !== "0") params.set("since_days", filters.sinceDays)
  if (filters.minSize !== "0") params.set("min_size", filters.minSize)
  if (filters.sort !== "newest") params.set("sort", filters.sort)
  return params.toString()
}

/**
 * 网格预览地址：图片走后端 `?w=` 缩略图（原图常见 5~10MB，铺满一页会拖垮首屏），
 * 视频没有缩略图接口，仍取原文件首帧。
 */
function previewUrl(item: MediaItem): string {
  return item.kind === "image" ? `${item.url}?w=480` : `${item.url}#t=0.1`
}

export default function MediaLibraryPage() {
  const [kind, setKind] = useState<MediaKind>("all")
  const [query, setQuery] = useState("")
  const [debouncedQuery, setDebouncedQuery] = useState("")
  const [sinceDays, setSinceDays] = useState("0")
  const [minSize, setMinSize] = useState("0")
  const [sortBy, setSortBy] = useState<SortMode>("newest")

  const [items, setItems] = useState<MediaItem[]>([])
  const [stats, setStats] = useState<MediaStats>(EMPTY_STATS)
  const [total, setTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selected, setSelected] = useState<string[]>([])
  // 全选/反选可能要翻页拉全量 id，期间给按钮一个忙碌态
  const [selectingAll, setSelectingAll] = useState(false)
  const [previewIndex, setPreviewIndex] = useState<number | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [reloadToken, setReloadToken] = useState(0)

  // 请求序号：切筛选或翻页时旧响应可能后到，靠序号丢弃过期结果（比 AbortController 更稳）
  const requestIdRef = useRef(0)

  const hasFilters =
    kind !== "all" || debouncedQuery !== "" || sinceDays !== "0" || minSize !== "0" || sortBy !== "newest"
  const allSelected = total > 0 && selected.length >= total

  // 搜索防抖：连续输入只在停顿 300ms 后发一次请求
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query.trim()), SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [query])

  // 换筛选条件等于换数据集：回到第一页并清空旧结果，否则新结果会接在旧列表后面
  useEffect(() => {
    setOffset(0)
    setItems([])
    setSelected([])
    setPreviewIndex(null)
  }, [kind, debouncedQuery, sinceDays, minSize, sortBy])

  useEffect(() => {
    const requestId = ++requestIdRef.current
    const controller = new AbortController()
    setLoading(true)
    setError(null)

    const queryString = buildMediaQuery(
      { kind, query: debouncedQuery, sinceDays, minSize, sort: sortBy },
      PAGE_SIZE,
      offset,
    )

    fetch(`${API_BASE}/api/admin/media?${queryString}`, {
      headers: getAuthHeader(),
      signal: controller.signal,
    })
      .then(async res => {
        const payload: unknown = await res.json().catch(() => ({}))
        if (!res.ok) throw new Error(responseDetail(payload, `HTTP ${res.status}`))
        return payload
      })
      .then(payload => {
        if (requestId !== requestIdRef.current) return
        const data = mediaListFrom(payload)
        // offset 为 0 是首屏/重新筛选，整表替换；否则是「加载更多」，追加
        setItems(prev => (offset === 0 ? data.items : [...prev, ...data.items]))
        setStats(data.stats)
        setTotal(data.total)
        setHasMore(data.has_more)
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted || requestId !== requestIdRef.current) return
        const message = errorMessage(err)
        setError(message)
        toast.error(`加载媒体失败: ${message.slice(0, 80)}`)
      })
      .finally(() => {
        if (requestId === requestIdRef.current) setLoading(false)
      })

    return () => controller.abort()
  }, [kind, debouncedQuery, sinceDays, minSize, sortBy, offset, reloadToken])

  const toggleSelect = (id: string) => {
    setSelected(prev => (prev.includes(id) ? prev.filter(value => value !== id) : [...prev, id]))
  }

  const handleRefresh = () => {
    setOffset(0)
    setReloadToken(token => token + 1)
  }

  const resetFilters = () => {
    setKind("all")
    setQuery("")
    setSinceDays("0")
    setMinSize("0")
    setSortBy("newest")
  }

  /**
   * 拉取「当前筛选条件下的全部 id」。
   *
   * 分页只加载了一部分时，全选/反选必须按整个筛选结果算，否则删不干净。
   * 100 条/页翻页，最多 SELECT_ALL_MAX_PAGES 页，超出就按已拿到的处理并提示。
   */
  const fetchAllIds = async (): Promise<string[]> => {
    const ids: string[] = []
    for (let page = 0; page < SELECT_ALL_MAX_PAGES; page += 1) {
      const queryString = buildMediaQuery(
        { kind, query: debouncedQuery, sinceDays, minSize, sort: sortBy },
        SELECT_ALL_PAGE_SIZE,
        page * SELECT_ALL_PAGE_SIZE,
      )
      const res = await fetch(`${API_BASE}/api/admin/media?${queryString}`, {
        headers: getAuthHeader(),
      })
      const payload: unknown = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(responseDetail(payload, `HTTP ${res.status}`))
      const data = mediaListFrom(payload)
      ids.push(...data.items.map(item => item.id))
      if (!data.has_more || data.items.length === 0) break
    }
    return ids
  }

  // 全选：已全选时再点一次等于取消；还有未加载的条目就去拉全量 id
  const handleSelectAll = async () => {
    if (selectingAll) return
    if (allSelected) {
      setSelected([])
      return
    }
    if (!hasMore) {
      setSelected(items.map(item => item.id))
      return
    }
    setSelectingAll(true)
    try {
      const ids = await fetchAllIds()
      setSelected(ids)
      if (ids.length < total) {
        toast.warning(`只选中了前 ${ids.length} 项（共 ${total} 项），请先缩小筛选范围`)
      }
    } catch (err: unknown) {
      toast.error(`全选失败: ${errorMessage(err)}`)
    } finally {
      setSelectingAll(false)
    }
  }

  // 反选：同样以「当前筛选结果全集」为准，和全选的选中范围保持一致
  const handleInvertSelection = async () => {
    if (selectingAll || total === 0) return
    setSelectingAll(true)
    try {
      const ids = hasMore ? await fetchAllIds() : items.map(item => item.id)
      const current = new Set(selected)
      setSelected(ids.filter(id => !current.has(id)))
    } catch (err: unknown) {
      toast.error(`反选失败: ${errorMessage(err)}`)
    } finally {
      setSelectingAll(false)
    }
  }

  const runDelete = async (ids: string[]) => {
    if (ids.length === 0 || deleting) return
    setDeleting(true)

    // 单个走 DELETE /{id}，多个走批删接口；两条路径的响应体不同，交给 deletedCount 归一
    const single = ids.length === 1
    try {
      const res = await fetch(
        single
          ? `${API_BASE}/api/admin/media/${encodeURIComponent(ids[0])}`
          : `${API_BASE}/api/admin/media/delete`,
        {
          method: single ? "DELETE" : "POST",
          headers: { "Content-Type": "application/json", ...getAuthHeader() },
          body: single ? undefined : JSON.stringify({ ids }),
        },
      )
      const payload: unknown = await res.json().catch(() => ({}))
      if (!res.ok) {
        const detail = responseDetail(payload, `HTTP ${res.status}`)
        toast.error(`删除失败: ${detail.slice(0, 80)}`)
        return
      }

      const removed = deletedCount(payload, ids.length)
      const idSet = new Set(ids)
      const removedItems = items.filter(item => idSet.has(item.id))
      const remaining = items.filter(item => !idSet.has(item.id))
      // 选中集合可能包含「还没加载出来」的条目（全选跨页），那时本地补丁算不准
      const hasUnloaded = ids.some(id => !items.some(item => item.id === id))

      setSelected(prev => prev.filter(id => !idSet.has(id)))
      setPreviewIndex(null)
      setDeleteTarget(null)
      toast.success(`已删除 ${removed} 项`)

      if (hasUnloaded || (remaining.length === 0 && hasMore)) {
        // 交给服务端重算：total / stats / 分页位置都以接口为准
        setItems([])
        setOffset(0)
        setReloadToken(token => token + 1)
        return
      }

      setItems(remaining)
      setTotal(prev => Math.max(0, prev - removed))
      // 统计行要跟着掉，否则「共 N 项」与「图片 x / 视频 y」会互相矛盾
      setStats(prev => ({
        image: Math.max(0, prev.image - removedItems.filter(item => item.kind === "image").length),
        video: Math.max(0, prev.video - removedItems.filter(item => item.kind === "video").length),
        bytes: Math.max(0, prev.bytes - removedItems.reduce((sum, item) => sum + item.size, 0)),
      }))
    } catch (err: unknown) {
      toast.error(`删除失败: ${errorMessage(err)}`)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <PageShell
      actions={
        <Button variant="outline" onClick={handleRefresh} disabled={loading}>
          <RefreshCw className={cn("mr-2 h-4 w-4", loading && "animate-spin")} /> 刷新
        </Button>
      }
    >
      <div className="space-y-5">
        <Card>
          <CardContent className="space-y-3">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <SegmentedGroup
                label="类型"
                value={kind}
                onChange={value => setKind(toKind(value))}
                options={KIND_OPTIONS}
              />
              <div className="relative w-full md:max-w-xs">
                <Search
                  className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden="true"
                />
                <Input
                  type="search"
                  value={query}
                  onChange={event => setQuery(event.target.value)}
                  placeholder="按提示词搜索…"
                  aria-label="按提示词搜索"
                  className="pl-8"
                />
              </div>
            </div>

            {/* 第二行：清理时最常用的三个维度（时间 / 体积 / 排序） */}
            <div className="flex flex-wrap items-end gap-3 border-t border-border pt-3">
              <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-xs text-muted-foreground sm:max-w-[11rem]">
                生成时间
                <Select
                  value={sinceDays}
                  onChange={event => setSinceDays(event.target.value)}
                  aria-label="按生成时间筛选"
                >
                  {SINCE_OPTIONS.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </Select>
              </label>
              <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-xs text-muted-foreground sm:max-w-[11rem]">
                体积
                <Select
                  value={minSize}
                  onChange={event => setMinSize(event.target.value)}
                  aria-label="按体积筛选"
                >
                  {MIN_SIZE_OPTIONS.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </Select>
              </label>
              <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-xs text-muted-foreground sm:max-w-[11rem]">
                排序
                <Select
                  value={sortBy}
                  onChange={event => setSortBy(toSort(event.target.value))}
                  aria-label="排序方式"
                >
                  {SORT_OPTIONS.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </Select>
              </label>
              {hasFilters && (
                <Button variant="ghost" size="sm" onClick={resetFilters} className="gap-1.5">
                  <RotateCcw className="h-3.5 w-3.5" /> 重置筛选
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* 统计 + 批量选择：选择操作常驻，不必先勾一条才能看到入口 */}
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted-foreground">
            <span className="tabular font-medium text-foreground">共 {total} 项</span>
            <span aria-hidden="true">·</span>
            <span className="tabular">图片 {stats.image}</span>
            <span aria-hidden="true">·</span>
            <span className="tabular">视频 {stats.video}</span>
            <span aria-hidden="true">·</span>
            <span className="tabular">占用 {formatBytes(stats.bytes)}</span>
            {selected.length > 0 && (
              <StatusBadge tone="accent" className="tabular">已选 {selected.length}</StatusBadge>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => void handleSelectAll()}
              disabled={selectingAll || total === 0}
              className="gap-1.5"
              aria-label={allSelected ? "取消全选" : "全选当前筛选结果"}
            >
              {selectingAll ? (
                <><RefreshCw className="h-3.5 w-3.5 animate-spin" /> 处理中…</>
              ) : allSelected ? (
                <><Square className="h-3.5 w-3.5" /> 取消全选</>
              ) : (
                <><CheckSquare className="h-3.5 w-3.5" /> {hasMore ? `全选全部 ${total} 项` : "全选"}</>
              )}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void handleInvertSelection()}
              disabled={selectingAll || total === 0}
            >
              反选
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setSelected([])}
              disabled={selected.length === 0}
            >
              清空选择
            </Button>
            {selected.length > 0 && (
              <Button
                size="sm"
                variant="destructive"
                onClick={() => setDeleteTarget({ ids: selected, label: `所选 ${selected.length} 项` })}
                className="gap-1.5"
              >
                <Trash2 className="h-3.5 w-3.5" /> 删除所选 ({selected.length})
              </Button>
            )}
          </div>
        </div>

        {error && <Notice tone="error">{error}</Notice>}

        {/* 首屏骨架：只在还没有任何数据时占位，翻页时保留已有卡片避免闪烁 */}
        {loading && items.length === 0 && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" aria-hidden="true">
            {Array.from({ length: 6 }, (_, index) => (
              <div key={index} className="animate-pulse rounded-xl border bg-muted/40 aspect-square" />
            ))}
          </div>
        )}

        {items.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {items.map((item, index) => {
              const checked = selected.includes(item.id)
              return (
                <Card key={item.id} className="group overflow-hidden">
                  <div className="relative aspect-square overflow-hidden bg-muted/40">
                    {/* 预览区整块是按钮：鼠标、触屏、键盘都能打开大图查看 */}
                    <button
                      type="button"
                      onClick={() => setPreviewIndex(index)}
                      aria-label={`放大查看第 ${index + 1} 项`}
                      className="block h-full w-full cursor-zoom-in focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                    >
                      {item.kind === "image" ? (
                        <img
                          src={previewUrl(item)}
                          alt={item.prompt}
                          loading="lazy"
                          decoding="async"
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        // #t=0.1 让浏览器直接抓首帧，否则封面是一片黑
                        <video
                          src={previewUrl(item)}
                          preload="metadata"
                          muted
                          playsInline
                          className="h-full w-full object-cover"
                        />
                      )}
                    </button>

                    {/* 选择框独立于预览按钮，勾选时不会顺带打开大图 */}
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleSelect(item.id)}
                      aria-label={`选择第 ${index + 1} 项${item.prompt ? `：${item.prompt}` : ""}`}
                      className="absolute left-2 top-2 h-4 w-4 cursor-pointer rounded border-input accent-primary"
                    />

                    {item.kind === "video" && (
                      <span className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-black/55 p-2 text-white">
                        <Play className="h-5 w-5" aria-hidden="true" />
                      </span>
                    )}

                    {/* 操作层：窄屏常驻可见（触屏没有 hover），桌面端 hover/聚焦才浮现；
                        容器不吃点击，按钮始终可点，避免「看不见但能误触」 */}
                    <div className="pointer-events-none absolute inset-0 opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100 sm:focus-within:opacity-100">
                      <button
                        type="button"
                        onClick={() => setPreviewIndex(index)}
                        aria-label="放大查看"
                        className="pointer-events-auto absolute right-2 top-2 rounded-md bg-black/60 p-1.5 text-white transition-colors hover:bg-black/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60"
                      >
                        <Maximize2 className="h-4 w-4" aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          setDeleteTarget({
                            ids: [item.id],
                            label: item.prompt ? `「${item.prompt.slice(0, 20)}」` : item.filename || "该项",
                          })
                        }
                        aria-label="删除"
                        className="pointer-events-auto absolute bottom-2 right-2 rounded-md bg-black/60 p-1.5 text-red-300 transition-colors hover:bg-red-500/80 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60"
                      >
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </div>
                  </div>

                  <CardContent className="space-y-2 py-3">
                    <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                      <StatusBadge tone={item.kind === "image" ? "accent" : "info"}>
                        {item.kind === "image" ? "图片" : "视频"}
                      </StatusBadge>
                      {item.ratio ? <span className="font-mono">{item.ratio}</span> : null}
                      {item.duration ? <span className="font-mono">{item.duration}s</span> : null}
                      <span className="tabular">{formatBytes(item.size)}</span>
                      <span className="tabular">{formatTimestamp(item.created_at)}</span>
                    </div>
                    <p className="line-clamp-2 text-xs text-muted-foreground">
                      {item.prompt || "（无提示词）"}
                    </p>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        )}

        {hasMore && (
          <div className="flex justify-center">
            <Button
              variant="outline"
              onClick={() => setOffset(items.length)}
              disabled={loading}
              className="gap-1.5"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" /> 加载中…
                </>
              ) : (
                "加载更多"
              )}
            </Button>
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <Card>
            <CardContent className="flex flex-col items-center gap-4 py-20 text-muted-foreground">
              <ImageIcon className="h-14 w-14 text-muted-foreground/20" aria-hidden="true" />
              <div className="text-center">
                <p className="font-medium text-foreground">
                  {hasFilters ? "当前筛选没有结果" : "还没有生成过媒体"}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {hasFilters
                    ? "换个条件试试，或直接重置筛选回到全部内容。"
                    : "生成的图片与视频会自动归档到这里，方便统一查看与清理。"}
                </p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
                {hasFilters ? (
                  <Button variant="outline" size="sm" onClick={resetFilters} className="gap-1.5">
                    <RotateCcw className="h-3.5 w-3.5" /> 重置筛选
                  </Button>
                ) : (
                  <>
                    <Button asChild variant="outline" size="sm">
                      <Link to="/images">去图片生成</Link>
                    </Button>
                    <Button asChild variant="outline" size="sm">
                      <Link to="/videos">去视频生成</Link>
                    </Button>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <Lightbox
        media={items.map(item => ({
          url: item.url,
          kind: item.kind,
          caption: item.prompt,
          downloadName: item.filename || undefined,
        }))}
        index={previewIndex}
        onIndexChange={setPreviewIndex}
        onClose={() => setPreviewIndex(null)}
      />

      {deleteTarget && (
        <ConfirmDialog
          title={`删除${deleteTarget.label}？`}
          description="删除后本地文件不可恢复，正在引用它的链接也会失效。"
          busy={deleting}
          onCancel={() => {
            if (!deleting) setDeleteTarget(null)
          }}
          onConfirm={() => void runDelete(deleteTarget.ids)}
        />
      )}
    </PageShell>
  )
}

/** 轻量确认弹窗：只有「删除」一个危险动作，没必要抽象成通用组件。 */
function ConfirmDialog({
  title,
  description,
  busy,
  onCancel,
  onConfirm,
}: {
  title: string
  description: string
  busy: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onCancel()
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [busy, onCancel])

  return (
    <div
      className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onClick={event => {
        if (event.target === event.currentTarget && !busy) onCancel()
      }}
    >
      <Card className="w-full max-w-sm">
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <h3 className="text-sm font-semibold text-foreground">{title}</h3>
            <p className="text-xs text-muted-foreground">{description}</p>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" size="sm" onClick={onCancel} disabled={busy}>
              取消
            </Button>
            <Button variant="destructive" size="sm" onClick={onConfirm} disabled={busy} className="gap-1.5">
              {busy ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" /> 删除中…
                </>
              ) : (
                "删除"
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
