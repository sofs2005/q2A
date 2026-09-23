import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "../components/ui/button"
import {
  Trash2,
  Plus,
  RefreshCw,
  Bot,
  ShieldCheck,
  MailWarning,
  X,
  Settings,
  Search,
  MoreHorizontal,
  SlidersHorizontal,
} from "lucide-react"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { checkRegisterUnlock } from "../lib/registerUnlock"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/card"
import { StatusBadge, type BadgeTone } from "../components/ui/badge"
import { Input, Field, Notice } from "../components/ui/field"
import { PageShell } from "../components/ui/page-shell"
import { cn } from "@/lib/utils"

type AccountItem = {
  email: string
  password?: string
  token?: string
  cookies?: string
  username?: string
  valid?: boolean
  inflight?: number
  rate_limited_until?: number
  activation_pending?: boolean
  status_code?: string
  status_text?: string
  last_error?: string
}

type PersonalizationMemory = {
  enable_memory: boolean
  enable_history_memory: boolean
}

type PersonalizationSettings = {
  memory: PersonalizationMemory
  tools_enabled: Record<string, boolean>
}

type PersonalizationTarget =
  | { kind: "single"; email: string }
  | { kind: "batch"; emails: string[] }

type StatusFilter = "all" | "valid" | "pending_activation" | "rate_limited" | "banned" | "disabled" | "invalid"

const CLEAR_CONFIRM_TEXT = "清空上游记录"
const PAGE_SIZE_OPTIONS = [10, 20, 50, 100] as const
const PERSONALIZATION_MODAL_TITLE_ID = "accounts-personalization-modal-title"
const PERSONALIZATION_MODAL_DESCRIPTION_ID = "accounts-personalization-modal-description"
const PERSONALIZATION_MODAL_CONFIRM_INPUT_ID = "accounts-personalization-confirm-input"
const PERSONALIZATION_MODAL_CONFIRM_HELP_ID = "accounts-personalization-confirm-help"

const PERSONALIZATION_TOOL_OPTIONS = [
  {
    key: "web_extractor",
    label: "网页提取",
    description: "专门用于访问指定的网页链接，并从中提取、总结或分析特定内容，忽略无关的页面元素。",
    defaultOn: false,
  },
  {
    key: "web_search_image",
    label: "图片搜索",
    description: "用于在互联网上查找与特定关键词相关的图片资源，返回图片、来源链接及描述信息等。",
    defaultOn: false,
  },
  {
    key: "web_search",
    label: "网络搜索",
    description: "用于在互联网上检索最新的文本信息、新闻、数据或特定知识，帮助用户获取实时的外部资讯。",
    defaultOn: false,
  },
  {
    key: "image_gen_tool",
    label: "图像生成",
    description: "根据用户的文字描述（提示词），从零开始创作并生成全新的、符合描述的图像。",
    defaultOn: true,
  },
  {
    key: "code_interpreter",
    label: "代码解释器",
    description: "一个内置的编程运行环境，可执行代码以进行复杂计算、数据分析、图表绘制或文件处理。",
    defaultOn: false,
  },
  {
    key: "history_retriever",
    label: "检索历史记忆",
    description: "用于在非当前的对话历史中快速查找、回顾或提取之前提及过的关键信息、上下文或用户指令。",
    defaultOn: false,
  },
  {
    key: "image_edit_tool",
    label: "图像编辑",
    description: "对现有图像进行修改操作，例如添加/移除物体、改变风格、调整局部细节或进行图像合成。",
    defaultOn: true,
  },
  {
    key: "bio",
    label: "更新记忆",
    description: "用于记录、更新或管理用户的个人偏好、关键事实及长期背景信息，确保 Qwen 在后续对话中能记住您的特定需求并保持上下文的一致性。",
    defaultOn: false,
  },
  {
    key: "image_zoom_in_tool",
    label: "图像局部放大",
    description: "用于对图像的特定区域进行高分辨率放大或聚焦，以便观察细节或为后续处理提供更清晰的局部视图。",
    defaultOn: true,
  },
] as const

function canClearChats(acc: AccountItem) {
  return Boolean(acc.cookies || acc.token)
}

/** 账号状态归一化：后端未给 status_code 时按 valid 兜底。 */
function normalizedStatus(acc: AccountItem): Exclude<StatusFilter, "all"> {
  switch (acc.status_code) {
    case "valid":
    case "pending_activation":
    case "rate_limited":
    case "banned":
    case "disabled":
      return acc.status_code
    default:
      return "invalid"
  }
}

function statusTone(code?: string): BadgeTone {
  switch (code) {
    case "valid":
      return "success"
    case "pending_activation":
      return "warning"
    case "rate_limited":
      return "warning"
    case "banned":
      return "danger"
    case "disabled":
      return "neutral"
    case "auth_error":
      return "neutral"
    default:
      return "danger"
  }
}

function statusText(acc: AccountItem) {
  switch (acc.status_code) {
    case "valid":
      return "可用"
    case "pending_activation":
      return "未激活"
    case "rate_limited":
      return "限流"
    case "banned":
      return "封禁"
    case "disabled":
      return "已禁用"
    case "auth_error":
      return "认证失效"
    default:
      return acc.valid ? "可用" : "失效"
  }
}

function localizeError(error?: string) {
  if (!error) return "未知错误"
  const lower = error.toLowerCase()
  if (lower.includes("activation already in progress")) return "账号正在激活中，请稍后刷新"
  if (lower.includes("activation link or token not found")) return "激活链接或 Token 获取失败"
  if (lower.includes("token invalid") || lower.includes("token") || lower.includes("auth")) return "Token 无效或认证失败"
  return error
}

function createDefaultPersonalizationSettings(): PersonalizationSettings {
  return {
    memory: {
      enable_memory: false,
      enable_history_memory: false,
    },
    tools_enabled: PERSONALIZATION_TOOL_OPTIONS.reduce((acc, option) => {
      acc[option.key] = option.defaultOn
      return acc
    }, {} as Record<string, boolean>),
  }
}

function buildPersonalizationPayload(settings: PersonalizationSettings): PersonalizationSettings {
  return {
    memory: {
      enable_memory: Boolean(settings.memory.enable_memory),
      enable_history_memory: Boolean(settings.memory.enable_history_memory),
    },
    tools_enabled: PERSONALIZATION_TOOL_OPTIONS.reduce((acc, option) => {
      acc[option.key] = Boolean(settings.tools_enabled[option.key] ?? option.defaultOn)
      return acc
    }, {} as Record<string, boolean>),
  }
}

async function readClearResponse(res: Response) {
  const contentType = res.headers.get("content-type") || ""
  const text = await res.text()

  if (!text.trim()) {
    return { data: null, rawText: "" }
  }

  if (contentType.includes("application/json")) {
    try {
      return { data: JSON.parse(text), rawText: text }
    } catch {
      return { data: null, rawText: text }
    }
  }

  try {
    return { data: JSON.parse(text), rawText: text }
  } catch {
    return { data: null, rawText: text }
  }
}

/** 行内低频操作（验证/激活/删除）收进带文字标签的「更多操作」菜单。 */
function RowMoreMenu({
  account,
  verifying,
  busy,
  showActivate,
  onVerify,
  onActivate,
  onDelete,
}: {
  account: AccountItem
  verifying: boolean
  busy: boolean
  showActivate: boolean
  onVerify: () => void
  onActivate: () => void
  onDelete: () => void
}) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return

    const handlePointerDown = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false)
    }

    document.addEventListener("mousedown", handlePointerDown)
    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("mousedown", handlePointerDown)
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [open])

  const run = (action: () => void) => {
    setOpen(false)
    action()
  }

  return (
    <div ref={containerRef} className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(prev => !prev)}
        disabled={busy}
        aria-haspopup="menu"
        aria-expanded={open}
        className="gap-1.5"
      >
        <MoreHorizontal className="h-4 w-4" /> 更多
      </Button>
      {open && (
        <div
          role="menu"
          aria-label={`${account.email} 的更多操作`}
          className="absolute right-0 z-30 mt-2 w-44 rounded-xl border border-border bg-popover p-1.5 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => run(onVerify)}
            disabled={verifying || busy}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {verifying
              ? <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />
              : <ShieldCheck className="h-4 w-4" />}
            验证账号
          </button>
          {showActivate && (
            <button
              type="button"
              role="menuitem"
              onClick={() => run(onActivate)}
              disabled={busy}
              className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-orange-700 transition-colors hover:bg-orange-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50 dark:text-orange-300"
            >
              <MailWarning className="h-4 w-4" /> 激活账号
            </button>
          )}
          <button
            type="button"
            role="menuitem"
            onClick={() => run(onDelete)}
            disabled={busy}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-destructive transition-colors hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Trash2 className="h-4 w-4" /> 删除账号
          </button>
        </div>
      )}
    </div>
  )
}

export default function AccountsPage() {
  const [accounts, setAccounts] = useState<AccountItem[]>([])
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [addFormOpen, setAddFormOpen] = useState(false)
  const [registering, setRegistering] = useState(false)
  const [registerUnlocked, setRegisterUnlocked] = useState(false)
  const [verifying, setVerifying] = useState<string | null>(null)
  const [verifyingAll, setVerifyingAll] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [selectedEmails, setSelectedEmails] = useState<string[]>([])
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState<(typeof PAGE_SIZE_OPTIONS)[number]>(10)
  const [query, setQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all")

  const [personalizationTarget, setPersonalizationTarget] = useState<PersonalizationTarget | null>(null)
  const [personalizationSettings, setPersonalizationSettings] = useState<PersonalizationSettings>(createDefaultPersonalizationSettings())
  const [personalizationPhrase, setPersonalizationPhrase] = useState("")
  const personalizationLoading = false
  const [personalizationSaving, setPersonalizationSaving] = useState(false)
  const [personalizationClearing, setPersonalizationClearing] = useState(false)
  const [statusChangingEmail, setStatusChangingEmail] = useState<string | null>(null)
  const [statusChangingAction, setStatusChangingAction] = useState<"disable" | "enable" | null>(null)
  const [statusBatchAction, setStatusBatchAction] = useState<"disable" | "enable" | null>(null)
  const pageSelectAllRef = useRef<HTMLInputElement | null>(null)
  const personalizationModalRef = useRef<HTMLDivElement | null>(null)
  const personalizationReturnFocusRef = useRef<HTMLElement | null>(null)

  // 邮箱+密码字段同时匹配时解锁注册功能
  useEffect(() => {
    let cancelled = false

    checkRegisterUnlock(email, password).then(unlocked => {
      if (!cancelled && unlocked) setRegisterUnlocked(true)
    })

    return () => {
      cancelled = true
    }
  }, [email, password])

  const fetchAccounts = () => {
    fetch(`${API_BASE}/api/admin/accounts`, { headers: getAuthHeader() })
      .then(res => {
        if (!res.ok) throw new Error("unauthorized")
        return res.json()
      })
      .then(data => {
        const nextAccounts = data.accounts || []
        setAccounts(nextAccounts)
        setSelectedEmails(prev => prev.filter(email => nextAccounts.some((acc: AccountItem) => acc.email === email)))
        setLoadFailed(false)
      })
      .catch(() => {
        setLoadFailed(true)
        toast.error("刷新账号列表失败，请检查会话密钥")
      })
      .finally(() => setLoaded(true))
  }

  useEffect(() => {
    fetchAccounts()
  }, [])

  const stats = useMemo(() => {
    const result = { valid: 0, pending: 0, rateLimited: 0, banned: 0, disabled: 0, invalid: 0 }
    for (const acc of accounts) {
      switch (acc.status_code) {
        case "valid":
          result.valid += 1
          break
        case "pending_activation":
          result.pending += 1
          break
        case "rate_limited":
          result.rateLimited += 1
          break
        case "banned":
          result.banned += 1
          break
        case "disabled":
          result.disabled += 1
          break
        default:
          result.invalid += 1
          break
      }
    }
    return result
  }, [accounts])

  // 搜索/筛选只作用于展示与可选集合；统计始终基于全部账号。
  const filteredAccounts = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    return accounts.filter(acc => {
      if (statusFilter !== "all" && normalizedStatus(acc) !== statusFilter) return false
      if (keyword && !acc.email.toLowerCase().includes(keyword)) return false
      return true
    })
  }, [accounts, query, statusFilter])

  const totalPages = Math.max(1, Math.ceil(filteredAccounts.length / pageSize))
  const safeCurrentPage = Math.min(currentPage, totalPages)
  const pageStartIndex = (safeCurrentPage - 1) * pageSize
  const pagedAccounts = useMemo(
    () => filteredAccounts.slice(pageStartIndex, pageStartIndex + pageSize),
    [filteredAccounts, pageStartIndex, pageSize],
  )
  const pageEndIndex = Math.min(pageStartIndex + pagedAccounts.length, filteredAccounts.length)
  const currentPageEmails = useMemo(
    () => pagedAccounts.map(acc => acc.email),
    [pagedAccounts],
  )
  const currentPageSelectedCount = currentPageEmails.filter(email => selectedEmails.includes(email)).length
  const isCurrentPageAllSelected = currentPageEmails.length > 0 && currentPageSelectedCount === currentPageEmails.length
  const isCurrentPagePartiallySelected = currentPageSelectedCount > 0 && !isCurrentPageAllSelected

  const clearableSelectedEmails = useMemo(() => {
    const clearableEmailSet = new Set(accounts.filter(canClearChats).map(acc => acc.email))
    return selectedEmails.filter(email => clearableEmailSet.has(email))
  }, [accounts, selectedEmails])
  const disabledSelectedEmails = useMemo(() => {
    const disabledEmailSet = new Set(accounts.filter(acc => acc.status_code === "disabled").map(acc => acc.email))
    return selectedEmails.filter(email => disabledEmailSet.has(email))
  }, [accounts, selectedEmails])
  const enabledSelectedEmails = useMemo(() => {
    const enabledEmailSet = new Set(accounts.filter(acc => acc.status_code !== "disabled").map(acc => acc.email))
    return selectedEmails.filter(email => enabledEmailSet.has(email))
  }, [accounts, selectedEmails])

  useEffect(() => {
    if (currentPage > totalPages) {
      setSelectedEmails([])
      setCurrentPage(totalPages)
    }
  }, [currentPage, totalPages])

  useEffect(() => {
    if (pageSelectAllRef.current) {
      pageSelectAllRef.current.indeterminate = isCurrentPagePartiallySelected
    }
  }, [isCurrentPagePartiallySelected])

  useEffect(() => {
    if (!personalizationTarget) return

    personalizationReturnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null

    return () => {
      personalizationReturnFocusRef.current?.focus()
      personalizationReturnFocusRef.current = null
    }
  }, [personalizationTarget])

  useEffect(() => {
    if (!personalizationTarget) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !(personalizationSaving || personalizationClearing)) {
        setPersonalizationTarget(null)
        setPersonalizationPhrase("")
        return
      }

      if (event.key !== "Tab") return

      const modal = personalizationModalRef.current
      if (!modal) return

      const focusableElements = Array.from(
        modal.querySelectorAll<HTMLElement>(
          'button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])',
        ),
      ).filter(element => !element.getAttribute("aria-hidden"))

      if (focusableElements.length === 0) {
        event.preventDefault()
        return
      }

      const firstElement = focusableElements[0]
      const lastElement = focusableElements[focusableElements.length - 1]
      const activeElement = document.activeElement

      if (event.shiftKey && (!activeElement || activeElement === firstElement || !modal.contains(activeElement))) {
        event.preventDefault()
        lastElement.focus()
      } else if (!event.shiftKey && activeElement === lastElement) {
        event.preventDefault()
        firstElement.focus()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [personalizationTarget, personalizationSaving, personalizationClearing])

  const toggleSelectedEmail = (email: string) => {
    setSelectedEmails(prev =>
      prev.includes(email) ? prev.filter(item => item !== email) : [...prev, email],
    )
  }

  const toggleCurrentPageSelection = () => {
    if (currentPageEmails.length === 0) return
    setSelectedEmails(isCurrentPageAllSelected ? [] : currentPageEmails)
  }

  const goToPage = (page: number) => {
    const nextPage = Math.min(Math.max(page, 1), totalPages)
    if (nextPage === safeCurrentPage) return
    setSelectedEmails([])
    setCurrentPage(nextPage)
  }

  const changePageSize = (nextPageSize: number) => {
    if (!PAGE_SIZE_OPTIONS.includes(nextPageSize as (typeof PAGE_SIZE_OPTIONS)[number])) return
    setSelectedEmails([])
    setCurrentPage(1)
    setPageSize(nextPageSize as (typeof PAGE_SIZE_OPTIONS)[number])
  }

  // 搜索/筛选变化时清空选择，避免被隐藏的账号成为批量操作对象。
  const applyQuery = (next: string) => {
    setQuery(next)
    setSelectedEmails([])
    setCurrentPage(1)
  }

  const applyStatusFilter = (next: StatusFilter) => {
    setStatusFilter(next)
    setSelectedEmails([])
    setCurrentPage(1)
  }

  const openBatchPersonalization = () => {
    if (clearableSelectedEmails.length === 0) return

    personalizationReturnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    setPersonalizationTarget({ kind: "batch", emails: clearableSelectedEmails })
    setPersonalizationSettings(createDefaultPersonalizationSettings())
    setPersonalizationPhrase("")
  }

  const openSinglePersonalization = (targetEmail: string) => {
    if (personalizationLoading || personalizationSaving || personalizationClearing) return

    personalizationReturnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    setPersonalizationTarget({ kind: "single", email: targetEmail })
    setPersonalizationSettings(createDefaultPersonalizationSettings())
    setPersonalizationPhrase("")
  }

  const closePersonalizationModal = () => {
    if (personalizationSaving || personalizationClearing) return
    setPersonalizationTarget(null)
    setPersonalizationPhrase("")
  }

  const runPersonalizationSave = async () => {
    if (!personalizationTarget || personalizationSaving || personalizationClearing) return

    const payload = buildPersonalizationPayload(personalizationSettings)
    const targetEmails = personalizationTarget.kind === "batch" ? personalizationTarget.emails : [personalizationTarget.email]

    if (targetEmails.length === 0) {
      toast.error("没有可保存的账号")
      return
    }

    setPersonalizationSaving(true)
    const id = toast.loading(personalizationTarget.kind === "batch" ? "正在保存所选账号个性化设置..." : `正在保存 ${personalizationTarget.email} 的个性化设置...`)

    try {
      const url =
        personalizationTarget.kind === "batch"
          ? `${API_BASE}/api/admin/accounts/personalization`
          : `${API_BASE}/api/admin/accounts/${encodeURIComponent(personalizationTarget.email)}/personalization`

      const body =
        personalizationTarget.kind === "batch"
          ? { emails: targetEmails, ...payload }
          : payload

      const res = await fetch(url, {
        method: "PUT",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
        body: JSON.stringify(body),
      })
      const { data, rawText } = await readClearResponse(res)

      if (!res.ok || (data && typeof data === "object" && data.ok === false)) {
        const errorMessage =
          (data && typeof data === "object" ? (data.detail || data.error || data.reason || data.message) : null) ||
          (rawText ? rawText.trim() : "") ||
          res.statusText ||
          "请求失败"
        throw new Error(localizeError(String(errorMessage)))
      }

      if (personalizationTarget.kind === "batch") {
        const summary = (data && typeof data === "object" ? data.summary : null) || {}
        const message = `批量保存完成：成功 ${summary.success || 0}，失败 ${summary.failed || 0}，跳过 ${summary.skipped || 0}`
        if ((summary.failed || 0) > 0 || (summary.skipped || 0) > 0) {
          toast.warning(message, { id, duration: 8000 })
        } else {
          toast.success(message, { id, duration: 8000 })
        }
      } else {
        toast.success(`已保存 ${personalizationTarget.email} 的个性化设置`, { id, duration: 8000 })
      }

      fetchAccounts()
      setPersonalizationTarget(null)
      setPersonalizationPhrase("")
    } catch (error) {
      const message = error instanceof Error ? error.message : (personalizationTarget.kind === "batch" ? "批量保存请求失败" : "保存请求失败")
      toast.error(message, { id, duration: 8000 })
    } finally {
      setPersonalizationSaving(false)
    }
  }

  const runPersonalizationClear = async () => {
    if (!personalizationTarget || personalizationPhrase !== CLEAR_CONFIRM_TEXT || personalizationClearing || personalizationSaving) return

    const batchEmails =
      personalizationTarget.kind === "batch"
        ? personalizationTarget.emails.filter(email => accounts.some(acc => acc.email === email && canClearChats(acc)))
        : []

    if (personalizationTarget.kind === "batch" && batchEmails.length === 0) {
      toast.error("所选账号缺少可用凭证，无法清理聊天记录")
      setPersonalizationTarget(null)
      setPersonalizationPhrase("")
      setSelectedEmails(clearableSelectedEmails)
      return
    }

    setPersonalizationClearing(true)
    const id = toast.loading(personalizationTarget.kind === "batch" ? "正在清理所选账号..." : `正在清理 ${personalizationTarget.email}...`)

    try {
      const url =
        personalizationTarget.kind === "batch"
          ? `${API_BASE}/api/admin/accounts/chats`
          : `${API_BASE}/api/admin/accounts/${encodeURIComponent(personalizationTarget.email)}/chats`

      const requestInit =
        personalizationTarget.kind === "batch"
          ? {
              method: "DELETE",
              headers: { "Content-Type": "application/json", ...getAuthHeader() },
              body: JSON.stringify({ emails: batchEmails }),
            }
          : {
              method: "DELETE",
              headers: getAuthHeader(),
            }

      const res = await fetch(url, requestInit)
      const { data, rawText } = await readClearResponse(res)

      if (!res.ok || (data && typeof data === "object" && data.ok === false)) {
        const errorMessage =
          (data && typeof data === "object" ? (data.detail || data.error || data.reason || data.message) : null) ||
          (rawText ? rawText.trim() : "") ||
          res.statusText ||
          "请求失败"
        throw new Error(localizeError(String(errorMessage)))
      }

      if (personalizationTarget.kind === "batch") {
        const summary = (data && typeof data === "object" ? data.summary : null) || {}
        toast.success(`批量清理完成：成功 ${summary.success || 0}，失败 ${summary.failed || 0}，跳过 ${summary.skipped || 0}`, { id, duration: 8000 })
        setSelectedEmails([])
      } else {
        if (data && typeof data === "object" && data.status === "success") {
          toast.success(`已清理 ${data.email}`, { id, duration: 8000 })
        } else if (data && typeof data === "object" && data.status === "skipped") {
          const reason = data.reason === "missing_credentials" ? "缺少可用凭证" : (data.reason || "已跳过")
          toast.warning(`${data.email}：${reason}`, { id, duration: 8000 })
        } else {
          toast.error(`清理失败：${localizeError((data && typeof data === "object" ? data.error || data.reason : undefined) || rawText)}`, { id, duration: 8000 })
        }
      }

      fetchAccounts()
      setPersonalizationTarget(null)
      setPersonalizationPhrase("")
    } catch (error) {
      const message = error instanceof Error ? error.message : (personalizationTarget.kind === "batch" ? "批量清理请求失败" : "清理请求失败")
      toast.error(message, { id, duration: 8000 })
    } finally {
      setPersonalizationClearing(false)
    }
  }

  const runAccountStatusChange = async (action: "disable" | "enable", targetEmail?: string) => {
    const isBatch = !targetEmail
    const targetEmails = isBatch ? selectedEmails : [targetEmail]
    const normalizedEmails = targetEmails.filter((email): email is string => Boolean(email))

    if (normalizedEmails.length === 0) {
      toast.error("没有可操作的账号")
      return
    }

    if (isBatch) {
      setStatusBatchAction(action)
    } else {
      setStatusChangingEmail(targetEmail ?? null)
      setStatusChangingAction(action)
    }

    const actionLabel = action === "disable" ? "禁用" : "启用"
    const id = toast.loading(isBatch ? `正在批量${actionLabel}所选账号...` : `正在${actionLabel} ${targetEmail}...`)

    try {
      const url = isBatch
        ? `${API_BASE}/api/admin/accounts/${action}`
        : `${API_BASE}/api/admin/accounts/${encodeURIComponent(targetEmail as string)}/${action}`
      const requestInit = isBatch
        ? {
            method: "POST",
            headers: { "Content-Type": "application/json", ...getAuthHeader() },
            body: JSON.stringify({ emails: normalizedEmails }),
          }
        : {
            method: "POST",
            headers: getAuthHeader(),
          }

      const res = await fetch(url, requestInit)
      const { data, rawText } = await readClearResponse(res)

      if (!res.ok || (data && typeof data === "object" && data.ok === false)) {
        const errorMessage =
          (data && typeof data === "object" ? (data.detail || data.error || data.reason || data.message) : null) ||
          (rawText ? rawText.trim() : "") ||
          res.statusText ||
          "请求失败"
        throw new Error(localizeError(String(errorMessage)))
      }

      if (isBatch) {
        const summary = (data && typeof data === "object" ? data.summary : null) || {}
        toast.success(`批量${actionLabel}完成：成功 ${summary.success || 0}，失败 ${summary.failed || 0}，跳过 ${summary.skipped || 0}`, { id, duration: 8000 })
        setSelectedEmails([])
      } else {
        toast.success(`已${actionLabel} ${targetEmail}`, { id, duration: 6000 })
      }

      fetchAccounts()
    } catch (error) {
      const message = error instanceof Error ? error.message : (isBatch ? `批量${actionLabel}请求失败` : `${actionLabel}请求失败`)
      toast.error(message, { id, duration: 8000 })
    } finally {
      if (isBatch) {
        setStatusBatchAction(null)
      } else {
        setStatusChangingEmail(null)
        setStatusChangingAction(null)
      }
    }
  }

  const handleAdd = () => {
    if (!email.trim() || !password.trim()) {
      toast.error("请填写邮箱和密码")
      return
    }
    const id = toast.loading("正在注入账号...")
    fetch(`${API_BASE}/api/admin/accounts`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ email, password })
    }).then(res => res.json())
      .then(data => {
        if (data.ok) {
          toast.success("账号已加入账号池", { id })
          setEmail("")
          setPassword("")
          setAddFormOpen(false)
          fetchAccounts()
        } else {
          toast.error(localizeError(data.error) || "账号注入失败", { id, duration: 8000 })
        }
      })
      .catch(() => toast.error("账号注入请求失败", { id }))
  }

  const handleDelete = (targetEmail: string) => {
    const id = toast.loading(`正在删除 ${targetEmail}...`)
    fetch(`${API_BASE}/api/admin/accounts/${encodeURIComponent(targetEmail)}`, {
      method: "DELETE",
      headers: getAuthHeader(),
    }).then(res => {
      if (!res.ok) throw new Error("delete failed")
      toast.success(`已删除 ${targetEmail}`, { id })
      fetchAccounts()
    }).catch(() => toast.error("删除账号失败", { id }))
  }

  const handleAutoRegister = () => {
    setRegistering(true)
    const id = toast.loading("正在自动注册新账号，请稍候...")
    fetch(`${API_BASE}/api/admin/accounts/register`, {
      method: "POST",
      headers: getAuthHeader(),
    }).then(res => res.json())
      .then(data => {
        if (data.activation_pending) {
          toast.warning(`账号已注册，但仍需激活：${data.email}`, { id, duration: 8000 })
          fetchAccounts()
        } else if (data.ok) {
          toast.success(data.message || `注册成功：${data.email}`, { id, duration: 8000 })
          fetchAccounts()
        } else {
          toast.error(localizeError(data.error) || "自动注册失败", { id, duration: 8000 })
          if (data.email) fetchAccounts()
        }
      })
      .catch(() => toast.error("自动注册请求失败", { id }))
      .finally(() => setRegistering(false))
  }

  const handleVerify = (targetEmail: string) => {
    setVerifying(targetEmail)
    const id = toast.loading(`正在验证 ${targetEmail}...`)
    fetch(`${API_BASE}/api/admin/accounts/${encodeURIComponent(targetEmail)}/verify`, {
      method: "POST",
      headers: getAuthHeader(),
    }).then(res => res.json())
      .then(data => {
        if (data.valid) {
          toast.success(`验证通过：${targetEmail}`, { id })
        } else {
          toast.error(`验证失败：${statusText(data) || localizeError(data.error)}`, { id, duration: 8000 })
        }
        fetchAccounts()
      })
      .catch(() => toast.error("验证请求失败", { id }))
      .finally(() => setVerifying(null))
  }

  const handleVerifyAll = () => {
    setVerifyingAll(true)
    const id = toast.loading("正在并发巡检所有账号...")
    fetch(`${API_BASE}/api/admin/verify`, {
      method: "POST",
      headers: getAuthHeader(),
    }).then(res => res.json())
      .then(data => {
        if (data.ok) {
          toast.success(`全量巡检完成，并发数：${data.concurrency || 1}`, { id })
        } else {
          toast.error("全量巡检失败", { id })
        }
        fetchAccounts()
      })
      .catch(() => toast.error("全量巡检请求失败", { id }))
      .finally(() => setVerifyingAll(false))
  }

  const handleActivate = (targetEmail: string) => {
    const id = toast.loading(`正在激活 ${targetEmail}...`)
    fetch(`${API_BASE}/api/admin/accounts/${encodeURIComponent(targetEmail)}/activate`, {
      method: "POST",
      headers: getAuthHeader(),
    }).then(res => res.json())
      .then(data => {
        if (data.pending) {
          toast.success(`账号正在激活中，请稍后刷新：${targetEmail}`, { id, duration: 6000 })
        } else if (data.ok) {
          toast.success(data.message || `激活成功：${targetEmail}`, { id, duration: 6000 })
        } else {
          toast.error(`激活失败：${localizeError(data.error || data.message)}`, { id, duration: 8000 })
        }
        fetchAccounts()
      })
      .catch(() => toast.error("激活请求失败", { id }))
  }

  const personalizationBusy = personalizationSaving || personalizationClearing
  const statusActionBusy = statusChangingEmail !== null || statusBatchAction !== null
  const actionsBusy = personalizationBusy || statusActionBusy
  const selectedCount = selectedEmails.length
  const filtersActive = query.trim() !== "" || statusFilter !== "all"

  const statChips: { key: StatusFilter; label: string; value: number; dot: string }[] = [
    { key: "valid", label: "可用", value: stats.valid, dot: "bg-emerald-500" },
    { key: "pending_activation", label: "未激活", value: stats.pending, dot: "bg-amber-500" },
    { key: "rate_limited", label: "限流", value: stats.rateLimited, dot: "bg-amber-500" },
    { key: "banned", label: "封禁", value: stats.banned, dot: "bg-red-500" },
    { key: "disabled", label: "已禁用", value: stats.disabled, dot: "bg-muted-foreground/40" },
    { key: "invalid", label: "其他失效", value: stats.invalid, dot: "bg-red-500" },
  ]

  const renderRowActions = (acc: AccountItem) => {
    const clearDisabled = !canClearChats(acc)
    const showActivate =
      acc.status_code !== "disabled" &&
      acc.status_code !== "valid" &&
      acc.status_code !== "rate_limited" &&
      acc.status_code !== "banned"

    return (
      <>
        <Button
          variant="outline"
          size="sm"
          onClick={() => openSinglePersonalization(acc.email)}
          disabled={personalizationBusy || personalizationLoading || actionsBusy || clearDisabled}
          title={clearDisabled ? "缺少 cookies 和 token，无法设置" : "管理该账号的设置"}
          className="gap-1.5"
        >
          <Settings className="h-4 w-4" /> 账号设置
        </Button>
        {acc.status_code === "disabled" ? (
          <Button
            variant="outline"
            size="sm"
            onClick={() => runAccountStatusChange("enable", acc.email)}
            disabled={actionsBusy}
            className="gap-1.5"
          >
            {statusChangingEmail === acc.email && statusChangingAction === "enable"
              ? <RefreshCw className="h-4 w-4 animate-spin" />
              : <ShieldCheck className="h-4 w-4" />}
            启用
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => runAccountStatusChange("disable", acc.email)}
            disabled={actionsBusy}
            className="gap-1.5"
          >
            {statusChangingEmail === acc.email && statusChangingAction === "disable"
              ? <RefreshCw className="h-4 w-4 animate-spin" />
              : <X className="h-4 w-4" />}
            禁用
          </Button>
        )}
        <RowMoreMenu
          account={acc}
          verifying={verifying === acc.email}
          busy={actionsBusy}
          showActivate={showActivate}
          onVerify={() => handleVerify(acc.email)}
          onActivate={() => handleActivate(acc.email)}
          onDelete={() => handleDelete(acc.email)}
        />
      </>
    )
  }

  return (
    <PageShell
      actions={
        <>
          <Button variant="outline" onClick={handleVerifyAll} disabled={verifyingAll}>
            <ShieldCheck className={cn("mr-2 h-4 w-4", verifyingAll && "animate-pulse")} /> 全量巡检
          </Button>
          <Button
            variant="outline"
            onClick={() => { fetchAccounts(); toast.success("账号列表已刷新") }}
          >
            <RefreshCw className="mr-2 h-4 w-4" /> 刷新状态
          </Button>
          {registerUnlocked && (
            <Button variant="outline" onClick={handleAutoRegister} disabled={registering}>
              {registering ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <Bot className="mr-2 h-4 w-4" />}
              {registering ? "正在注册..." : "一键获取新号"}
            </Button>
          )}
          <Button onClick={() => setAddFormOpen(prev => !prev)} aria-expanded={addFormOpen}>
            <Plus className="mr-2 h-4 w-4" /> 添加账号
          </Button>
        </>
      }
    >
      <div className="space-y-6">
        {loadFailed && (
          <Notice tone="error">
            无法读取账号列表。请在「系统设置」中确认当前会话 Key 是否正确。
          </Notice>
        )}

        {/* 添加账号：默认折叠，点主按钮后就地展开聚焦表单 */}
        {addFormOpen && (
          <Card>
            <CardHeader>
              <CardTitle>添加账号</CardTitle>
              <CardDescription>填写账号邮箱和密码，系统将自动登录并获取 Token。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-4 md:flex-row md:items-end">
                <Field label="邮箱" className="w-full md:flex-1">
                  <Input
                    type="text"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="账号邮箱地址"
                    autoComplete="off"
                    autoFocus
                  />
                </Field>
                <Field label="密码" className="w-full md:flex-1">
                  <Input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="账号密码"
                    autoComplete="new-password"
                  />
                </Field>
                <div className="flex gap-2">
                  <Button variant="ghost" onClick={() => setAddFormOpen(false)} className="h-10">
                    取消
                  </Button>
                  <Button onClick={handleAdd} className="h-10 md:w-auto">
                    <Plus className="mr-2 h-4 w-4" /> 确认添加
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 账号池摘要：状态即筛选入口 */}
        <section className="space-y-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold text-foreground">账号池摘要</h3>
            <p className="text-xs text-muted-foreground">
              共 <span className="tabular font-medium text-foreground">{accounts.length}</span> 个账号
              {filtersActive ? (
                <>
                  ，筛选后 <span className="tabular font-medium text-foreground">{filteredAccounts.length}</span> 个
                </>
              ) : null}
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-7">
            <button
              type="button"
              onClick={() => applyStatusFilter("all")}
              aria-pressed={statusFilter === "all"}
              className={cn(
                "rounded-xl border px-4 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                statusFilter === "all"
                  ? "border-primary bg-primary/10"
                  : "border-border bg-card hover:border-foreground/20 hover:bg-accent/40",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-muted-foreground">全部</span>
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden="true" />
              </div>
              <div className="tabular mt-1.5 text-2xl font-semibold tracking-tight">{accounts.length}</div>
            </button>
            {statChips.map(stat => (
              <button
                key={stat.key}
                type="button"
                onClick={() => applyStatusFilter(statusFilter === stat.key ? "all" : stat.key)}
                aria-pressed={statusFilter === stat.key}
                className={cn(
                  "rounded-xl border px-4 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                  statusFilter === stat.key
                    ? "border-primary bg-primary/10"
                    : "border-border bg-card hover:border-foreground/20 hover:bg-accent/40",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
                  <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", stat.dot)} aria-hidden="true" />
                </div>
                <div className="tabular mt-1.5 text-2xl font-semibold tracking-tight">{stat.value}</div>
              </button>
            ))}
          </div>
        </section>

        {/* 选择后出现的批量操作栏：全部是有文字标签的显式按钮 */}
        {selectedCount > 0 && (
          <div className="sticky top-[4.5rem] z-10 flex flex-wrap items-center gap-2 rounded-xl border border-primary/30 bg-primary/5 px-4 py-3 backdrop-blur">
            <span className="mr-1 text-sm">
              已选 <span className="tabular font-semibold text-foreground">{selectedCount}</span> 个账号
            </span>
            <Button
              size="sm"
              variant="outline"
              onClick={() => runAccountStatusChange("enable")}
              disabled={actionsBusy || disabledSelectedEmails.length === 0}
              title={disabledSelectedEmails.length > 0 ? "批量启用所选账号" : "所选账号中没有已禁用账号"}
              className="gap-1.5"
            >
              <ShieldCheck className="h-4 w-4" /> 批量启用 ({disabledSelectedEmails.length})
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => runAccountStatusChange("disable")}
              disabled={actionsBusy || enabledSelectedEmails.length === 0}
              title={enabledSelectedEmails.length > 0 ? "批量禁用所选账号" : "所选账号中没有未禁用账号"}
              className="gap-1.5"
            >
              <X className="h-4 w-4" /> 批量禁用 ({enabledSelectedEmails.length})
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={openBatchPersonalization}
              disabled={personalizationLoading || actionsBusy || clearableSelectedEmails.length === 0}
              title={clearableSelectedEmails.length > 0 ? "管理所选账号的设置" : "请先勾选可设置的账号"}
              className="gap-1.5"
            >
              <Settings className="h-4 w-4" /> 批量账号设置 ({clearableSelectedEmails.length})
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setSelectedEmails([])}>
              取消选择
            </Button>
          </div>
        )}

        {/* 账号工作列表：桌面与手机共用同一套行结构，避免窄屏横向裁切 */}
        <Card>
          <div className="flex flex-col gap-3 border-b border-border bg-muted/40 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-2">
              <CardTitle>账号列表</CardTitle>
              <StatusBadge tone="accent" className="tabular">{filteredAccounts.length}</StatusBadge>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div className="relative min-w-0 sm:w-64">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
                <Input
                  type="search"
                  value={query}
                  onChange={e => applyQuery(e.target.value)}
                  placeholder="搜索账号邮箱"
                  aria-label="搜索账号邮箱"
                  className="h-9 pl-8"
                />
              </div>
              <div className="flex items-center gap-2">
                <label htmlFor="accounts-status-filter" className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" /> 状态
                </label>
                <select
                  id="accounts-status-filter"
                  value={statusFilter}
                  onChange={event => applyStatusFilter(event.target.value as StatusFilter)}
                  className="h-9 rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                >
                  <option value="all">全部</option>
                  <option value="valid">可用</option>
                  <option value="pending_activation">未激活</option>
                  <option value="rate_limited">限流</option>
                  <option value="banned">封禁</option>
                  <option value="disabled">已禁用</option>
                  <option value="invalid">其他失效</option>
                </select>
              </div>
            </div>
          </div>

          {pagedAccounts.length > 0 && (
            <div className="flex items-center gap-3 border-b border-border px-5 py-2.5">
              <input
                ref={pageSelectAllRef}
                type="checkbox"
                checked={isCurrentPageAllSelected}
                onChange={toggleCurrentPageSelection}
                aria-label="选择当前页账号"
                disabled={currentPageEmails.length === 0 || actionsBusy}
                className="h-4 w-4 rounded border-input accent-[hsl(var(--primary))]"
              />
              <span className="text-xs text-muted-foreground">全选当前页</span>
            </div>
          )}

          <ul className="divide-y divide-border">
            {filteredAccounts.length === 0 && (
              <li className="px-5 py-14 text-center text-sm text-muted-foreground">
                {!loaded
                  ? "正在加载账号列表…"
                  : filtersActive
                    ? "没有符合当前搜索/筛选条件的账号。"
                    : "暂无账号，请手动注入或一键获取新号。"}
              </li>
            )}
            {pagedAccounts.map((acc, index) => {
              const rowNumber = pageStartIndex + index + 1

              return (
                <li
                  key={acc.email}
                  className={cn(
                    "flex flex-col gap-3 px-5 py-3.5 transition-colors hover:bg-muted/40 lg:flex-row lg:items-center lg:gap-4",
                    selectedEmails.includes(acc.email) && "bg-primary/5",
                  )}
                >
                  <div className="flex min-w-0 flex-1 items-start gap-3">
                    <input
                      type="checkbox"
                      checked={selectedEmails.includes(acc.email)}
                      onChange={() => toggleSelectedEmail(acc.email)}
                      aria-label={`选择 ${acc.email}`}
                      disabled={actionsBusy}
                      className="mt-0.5 h-4 w-4 shrink-0 rounded border-input accent-[hsl(var(--primary))]"
                    />
                    <span className="tabular mt-0.5 shrink-0 font-mono text-xs text-muted-foreground">
                      #{rowNumber}
                    </span>
                    <div className="min-w-0 flex-1 space-y-1.5">
                      <p className="break-all font-mono text-sm text-foreground">{acc.email}</p>
                      <div className="flex flex-wrap items-center gap-2">
                        <StatusBadge tone={statusTone(acc.status_code)}>{statusText(acc)}</StatusBadge>
                        <StatusBadge tone="neutral" className="tabular font-mono">
                          {acc.inflight || 0} 线程
                        </StatusBadge>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 lg:shrink-0 lg:justify-end">
                    {renderRowActions(acc)}
                  </div>
                </li>
              )
            })}
          </ul>

          <div className="flex flex-col gap-3 border-t border-border bg-muted/30 px-5 py-4 text-sm text-muted-foreground md:flex-row md:items-center md:justify-between">
            <div className="tabular">
              {filteredAccounts.length > 0
                ? `显示第 ${pageStartIndex + 1}-${pageEndIndex} 条，共 ${filteredAccounts.length} 条`
                : "共 0 条账号"}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2">
                <span>每页</span>
                <select
                  value={pageSize}
                  onChange={event => changePageSize(Number(event.target.value))}
                  className="h-9 rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                  aria-label="每页账号数量"
                >
                  {PAGE_SIZE_OPTIONS.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
                <span>条</span>
              </label>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => goToPage(safeCurrentPage - 1)} disabled={safeCurrentPage <= 1}>
                  上一页
                </Button>
                <span className="tabular min-w-20 text-center text-foreground">
                  {safeCurrentPage} / {totalPages}
                </span>
                <Button variant="outline" size="sm" onClick={() => goToPage(safeCurrentPage + 1)} disabled={safeCurrentPage >= totalPages}>
                  下一页
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* 空筛选提示 */}
        {filtersActive && filteredAccounts.length === 0 && accounts.length > 0 && (
          <div className="flex justify-center">
            <Button
              variant="outline"
              onClick={() => { applyQuery(""); applyStatusFilter("all") }}
              className="gap-1.5"
            >
              <X className="h-4 w-4" /> 清除搜索与筛选
            </Button>
          </div>
        )}

        {personalizationTarget && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4 py-6" onClick={closePersonalizationModal}>
            <div
              ref={personalizationModalRef}
              className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl border bg-background p-5 shadow-xl sm:p-6"
              onClick={e => e.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby={PERSONALIZATION_MODAL_TITLE_ID}
              aria-describedby={PERSONALIZATION_MODAL_DESCRIPTION_ID}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h4 id={PERSONALIZATION_MODAL_TITLE_ID} className="text-base font-semibold">
                    {personalizationTarget.kind === "batch"
                      ? `管理所选 ${personalizationTarget.emails.length} 个账号的个性化设置`
                      : `管理 ${personalizationTarget.email} 的个性化设置`}
                  </h4>
                  <p id={PERSONALIZATION_MODAL_DESCRIPTION_ID} className="mt-1 text-sm text-muted-foreground">
                    {personalizationTarget.kind === "batch"
                      ? `当前选中 ${personalizationTarget.emails.length} 个账号，保存后会把同一份设置应用到这些账号。`
                      : `目标账号：${personalizationTarget.email}`}
                  </p>
                </div>
                <Button variant="ghost" size="icon" onClick={closePersonalizationModal} disabled={personalizationBusy} aria-label="关闭个性化设置弹窗">
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <div className="mt-5 space-y-4">
                <section className="rounded-xl border bg-muted/30 p-4">
                  <h5 className="text-sm font-semibold">记忆设置</h5>
                  <div className="mt-3 space-y-2">
                    <label className="flex items-center justify-between gap-4 rounded-lg border bg-background px-3 py-2 text-sm">
                      <span>
                        <span className="font-medium">启用记忆</span>
                        <span className="ml-2 text-xs text-muted-foreground">让账号保留长期记忆</span>
                      </span>
                      <input
                        type="checkbox"
                        checked={personalizationSettings.memory.enable_memory}
                        onChange={e => setPersonalizationSettings(prev => ({
                          ...prev,
                          memory: {
                            ...prev.memory,
                            enable_memory: e.target.checked,
                          },
                        }))}
                        disabled={personalizationBusy}
                        className="h-4 w-4 shrink-0 rounded border-input accent-[hsl(var(--primary))]"
                      />
                    </label>
                    <label className="flex items-center justify-between gap-4 rounded-lg border bg-background px-3 py-2 text-sm">
                      <span>
                        <span className="font-medium">启用历史记忆</span>
                        <span className="ml-2 text-xs text-muted-foreground">让账号保留历史上下文记忆</span>
                      </span>
                      <input
                        type="checkbox"
                        checked={personalizationSettings.memory.enable_history_memory}
                        onChange={e => setPersonalizationSettings(prev => ({
                          ...prev,
                          memory: {
                            ...prev.memory,
                            enable_history_memory: e.target.checked,
                          },
                        }))}
                        disabled={personalizationBusy}
                        className="h-4 w-4 shrink-0 rounded border-input accent-[hsl(var(--primary))]"
                      />
                    </label>
                  </div>
                </section>

                <section className="rounded-xl border bg-muted/30 p-4">
                  <h5 className="text-sm font-semibold">工具设置</h5>
                  <p className="mt-1 text-xs text-muted-foreground">共 9 个工具开关，保存时会按当前勾选状态同步到目标账号。</p>
                  <div className="mt-3 grid gap-2 md:grid-cols-2">
                    {PERSONALIZATION_TOOL_OPTIONS.map(option => (
                      <label key={option.key} className="flex items-start justify-between gap-4 rounded-lg border bg-background px-3 py-2 text-sm">
                        <span className="min-w-0">
                          <span className="block font-medium">{option.label}</span>
                          <span className="block text-xs text-muted-foreground">{option.description}</span>
                          <span className="block truncate font-mono text-[11px] text-muted-foreground/70">{option.key}</span>
                        </span>
                        <input
                          type="checkbox"
                          checked={Boolean(personalizationSettings.tools_enabled[option.key])}
                          onChange={e => setPersonalizationSettings(prev => ({
                            ...prev,
                            tools_enabled: {
                              ...prev.tools_enabled,
                              [option.key]: e.target.checked,
                            },
                          }))}
                          disabled={personalizationBusy}
                          className="mt-0.5 h-4 w-4 shrink-0 rounded border-input accent-[hsl(var(--primary))]"
                        />
                      </label>
                    ))}
                  </div>
                </section>

                <section className="rounded-xl border border-destructive/25 bg-destructive/5 p-4">
                  <h5 className="text-sm font-semibold text-destructive">清空上游记录</h5>
                  <p className="mt-1 text-xs text-destructive/80">
                    {personalizationTarget.kind === "batch"
                      ? `如需清理所选 ${personalizationTarget.emails.length} 个账号的上游聊天记录，请输入确认短语。`
                      : `如需清理 ${personalizationTarget.email} 的上游聊天记录，请输入确认短语。`}
                  </p>
                  <label
                    htmlFor={PERSONALIZATION_MODAL_CONFIRM_INPUT_ID}
                    id={PERSONALIZATION_MODAL_CONFIRM_HELP_ID}
                    className="mt-3 block text-sm font-medium text-destructive"
                  >
                    {`请输入「${CLEAR_CONFIRM_TEXT}」以确认执行清理操作。`}
                  </label>
                  <input
                    id={PERSONALIZATION_MODAL_CONFIRM_INPUT_ID}
                    autoFocus
                    value={personalizationPhrase}
                    onChange={e => setPersonalizationPhrase(e.target.value)}
                    disabled={personalizationBusy}
                    className="mt-3 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                    placeholder={CLEAR_CONFIRM_TEXT}
                    aria-describedby={PERSONALIZATION_MODAL_CONFIRM_HELP_ID}
                  />
                </section>
              </div>

              <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:items-center sm:justify-between">
                <Button
                  variant="destructive"
                  onClick={runPersonalizationClear}
                  disabled={personalizationBusy || personalizationPhrase !== CLEAR_CONFIRM_TEXT}
                >
                  {personalizationClearing ? "清理中..." : "确认清理"}
                </Button>
                <div className="flex gap-2 sm:justify-end">
                  <Button variant="outline" onClick={closePersonalizationModal} disabled={personalizationBusy}>
                    取消
                  </Button>
                  <Button variant="default" onClick={runPersonalizationSave} disabled={personalizationBusy || personalizationLoading}>
                    {personalizationSaving ? "保存中..." : (personalizationTarget.kind === "batch" ? "批量保存" : "保存设置")}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </PageShell>
  )
}
