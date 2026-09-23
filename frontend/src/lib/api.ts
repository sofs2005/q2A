/**
 * 后端 API 基础地址。
 *
 * - 本地开发：留空，由 Vite proxy 代理到 http://localhost:7860
 * - Docker 生产：留空，由 nginx proxy 代理到 backend:7860
 * - Vercel / 独立前端：设置 VITE_API_BASE_URL=https://your-backend.example.com
 */
export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string) ?? ''

/**
 * 从 catch 到的未知值里取错误消息。
 *
 * catch 子句的类型是 unknown，直接读 .message 会触发 no-explicit-any；
 * 非 Error（例如抛出的字符串）或消息为空时回退到 fallback，语义与原先的
 * `err.message || "网络错误"` 一致。
 */
export function errorMessage(err: unknown, fallback = "网络错误"): string {
  return (err instanceof Error && err.message) || fallback
}

/** 从响应体中取出可读的错误文案，兼容 detail（FastAPI）与 error（OpenAI 风格）。 */
export function responseDetail(payload: unknown, fallback: string): string {
  const { detail, error } = asRecord(payload)
  return String(detail || error || fallback)
}

/** 取响应体里的 `data` 数组；缺失或非数组时返回空数组。 */
export function responseItems<T>(payload: unknown): T[] {
  const data = asRecord(payload).data
  return Array.isArray(data) ? (data as T[]) : []
}

/** 把 unknown 收窄成可安全取属性的对象；非对象返回空对象。 */
function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {}
}

/**
 * 从 SSE data 事件里取出错误文案，兼容 error 为字符串或 { message } 两种形态。
 * 无 error 字段时返回 null（表示该事件是正常结果，不是错误）。
 */
export function sseErrorDetail(payload: unknown, fallback = "生成失败"): string | null {
  const { error } = asRecord(payload)
  if (!error) return null
  const message = typeof error === "object" ? asRecord(error).message : error
  return String(message || fallback)
}

/** 图片生成响应条目（仅声明前端实际读取的字段）。 */
export interface ImageResponseItem {
  url: string
  revised_prompt?: string
}

/** 视频生成响应条目（仅声明前端实际读取的字段）。 */
export interface VideoResponseItem {
  url: string
  revised_prompt?: string
  ratio?: string
  duration?: number
}
