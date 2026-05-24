import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "../components/ui/button"
import { Trash2, Plus, RefreshCw, Bot, ShieldCheck, MailWarning, X } from "lucide-react"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { checkRegisterUnlock } from "../lib/registerUnlock"

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

function statusStyle(code?: string) {
  switch (code) {
    case "valid":
      return "bg-green-500/10 text-green-700 dark:text-green-400 ring-green-500/20"
    case "pending_activation":
      return "bg-orange-500/10 text-orange-700 dark:text-orange-400 ring-orange-500/20"
    case "rate_limited":
      return "bg-yellow-500/10 text-yellow-700 dark:text-yellow-300 ring-yellow-500/20"
    case "banned":
      return "bg-red-500/10 text-red-700 dark:text-red-400 ring-red-500/20"
    case "auth_error":
      return "bg-slate-500/10 text-slate-700 dark:text-slate-300 ring-slate-500/20"
    default:
      return "bg-red-500/10 text-red-700 dark:text-red-400 ring-red-500/20"
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

export default function AccountsPage() {
  const [accounts, setAccounts] = useState<AccountItem[]>([])
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [token, setToken] = useState("")
  const [registering, setRegistering] = useState(false)
  const [registerUnlocked, setRegisterUnlocked] = useState(false)
  const [verifying, setVerifying] = useState<string | null>(null)
  const [verifyingAll, setVerifyingAll] = useState(false)
  const [selectedEmails, setSelectedEmails] = useState<string[]>([])
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState<(typeof PAGE_SIZE_OPTIONS)[number]>(10)

  const [personalizationTarget, setPersonalizationTarget] = useState<PersonalizationTarget | null>(null)
  const [personalizationSettings, setPersonalizationSettings] = useState<PersonalizationSettings>(createDefaultPersonalizationSettings())
  const [personalizationPhrase, setPersonalizationPhrase] = useState("")
  const personalizationLoading = false
  const [personalizationSaving, setPersonalizationSaving] = useState(false)
  const [personalizationClearing, setPersonalizationClearing] = useState(false)
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
        setSelectedEmails(prev => prev.filter(email => nextAccounts.some((acc: AccountItem) => acc.email === email && canClearChats(acc))))
      })
      .catch(() => toast.error("刷新账号列表失败，请检查会话密钥"))
  }

  useEffect(() => {
    fetchAccounts()
  }, [])

  const stats = useMemo(() => {
    const result = { valid: 0, pending: 0, rateLimited: 0, banned: 0, invalid: 0 }
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
        default:
          result.invalid += 1
          break
      }
    }
    return result
  }, [accounts])

  const totalPages = Math.max(1, Math.ceil(accounts.length / pageSize))
  const safeCurrentPage = Math.min(currentPage, totalPages)
  const pageStartIndex = (safeCurrentPage - 1) * pageSize
  const pagedAccounts = useMemo(
    () => accounts.slice(pageStartIndex, pageStartIndex + pageSize),
    [accounts, pageStartIndex, pageSize],
  )
  const pageEndIndex = Math.min(pageStartIndex + pagedAccounts.length, accounts.length)
  const currentPageClearableEmails = useMemo(
    () => pagedAccounts.filter(canClearChats).map(acc => acc.email),
    [pagedAccounts],
  )
  const currentPageSelectedCount = currentPageClearableEmails.filter(email => selectedEmails.includes(email)).length
  const isCurrentPageAllSelected = currentPageClearableEmails.length > 0 && currentPageSelectedCount === currentPageClearableEmails.length
  const isCurrentPagePartiallySelected = currentPageSelectedCount > 0 && !isCurrentPageAllSelected

  const clearableSelectedEmails = useMemo(() => {
    const clearableEmailSet = new Set(accounts.filter(canClearChats).map(acc => acc.email))
    return selectedEmails.filter(email => clearableEmailSet.has(email))
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
    if (currentPageClearableEmails.length === 0) return
    setSelectedEmails(isCurrentPageAllSelected ? [] : currentPageClearableEmails)
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

  const handleAdd = () => {
    if (!token.trim()) {
      toast.error("请先填写 Token")
      return
    }
    const id = toast.loading("正在注入账号...")
    fetch(`${API_BASE}/api/admin/accounts`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({
        email: email || `manual_${Date.now()}@qwen`,
        password,
        token,
      })
    }).then(res => res.json())
      .then(data => {
        if (data.ok) {
          toast.success("账号已加入账号池", { id })
          setEmail("")
          setPassword("")
          setToken("")
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

  return (
    <div className="space-y-6 relative">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-extrabold tracking-tight">{"账号管理"}</h2>
          <p className="text-muted-foreground mt-1">{"统一管理上游账号池，并区分未激活、限流、封禁与失效状态。"}</p>
        </div>
        <div className="flex gap-2 flex-wrap justify-end">
          <Button variant="secondary" onClick={handleVerifyAll} disabled={verifyingAll}>
            <ShieldCheck className={`mr-2 h-4 w-4 ${verifyingAll ? 'animate-pulse' : ''}`} /> {"全量巡检"}
          </Button>
          <Button
            variant="outline"
            onClick={openBatchPersonalization}
            disabled={personalizationLoading || personalizationBusy || clearableSelectedEmails.length === 0}
            className="border-red-500/30 text-red-600 hover:bg-red-500/10 hover:text-red-700 dark:text-red-400"
            title={clearableSelectedEmails.length > 0 ? "管理所选账号的设置" : "请先勾选可设置的账号"}
          >
            <Trash2 className="mr-2 h-4 w-4" /> {`批量账号设置 (${clearableSelectedEmails.length})`}
          </Button>
          <Button variant="outline" onClick={() => { fetchAccounts(); toast.success("账号列表已刷新") }}>
            <RefreshCw className="mr-2 h-4 w-4" /> {"刷新状态"}
          </Button>
          {registerUnlocked && (
            <Button variant="default" onClick={handleAutoRegister} disabled={registering}>
              {registering ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <Bot className="mr-2 h-4 w-4" />}
              {registering ? "正在注册..." : "一键获取新号"}
            </Button>
          )}
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-5">
        <div className="rounded-xl border bg-card p-4"><div className="text-sm text-muted-foreground">{"可用"}</div><div className="text-2xl font-bold">{stats.valid}</div></div>
        <div className="rounded-xl border bg-card p-4"><div className="text-sm text-muted-foreground">{"未激活"}</div><div className="text-2xl font-bold">{stats.pending}</div></div>
        <div className="rounded-xl border bg-card p-4"><div className="text-sm text-muted-foreground">{"限流"}</div><div className="text-2xl font-bold">{stats.rateLimited}</div></div>
        <div className="rounded-xl border bg-card p-4"><div className="text-sm text-muted-foreground">{"封禁"}</div><div className="text-2xl font-bold">{stats.banned}</div></div>
        <div className="rounded-xl border bg-card p-4"><div className="text-sm text-muted-foreground">{"其他失效"}</div><div className="text-2xl font-bold">{stats.invalid}</div></div>
      </div>

      <div className="rounded-2xl border bg-card/40 p-6 space-y-4">
        <div>
          <h3 className="text-base font-bold">{"手动注入账号"}</h3>
          <p className="text-sm text-muted-foreground">{"请先在 chat.qwen.ai 登录，然后按 F12 打开开发者工具，在 Application / Storage 里的 Local Storage / 本地存储 中找到 token 并直接复制完整原始值粘贴到下方输入框。"}</p>
          <div className="rounded-xl border border-orange-500/30 bg-orange-500/10 p-3 mt-3">
            <p className="text-sm font-semibold text-orange-700 dark:text-orange-300">{"重要：请只粘贴 Local Storage / 本地存储 里的 token 原始值，不要从 Network 请求或 Authorization 请求头中提取。"}</p>
            <p className="text-xs text-orange-700/80 dark:text-orange-200/80 mt-1">{"请不要带 Bearer 前缀，也不要粘贴整段 Authorization 文本。邮箱和密码可以不填，系统会在注入前先验证 token 是否有效。"}</p>
          </div>
        </div>
        <div className="flex flex-col md:flex-row gap-4 items-end">
          <div className="flex-1 w-full">
            <label className="text-xs font-semibold mb-1.5 block">{"Token（必填）"}</label>
            <input type="text" value={token} onChange={e => setToken(e.target.value)} className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm" placeholder={"粘贴从 Local Storage / 本地存储 直接复制的 token"} />
          </div>
          <div className="w-full md:w-64">
            <label className="text-xs font-semibold mb-1.5 block">{"邮箱（选填）"}</label>
            <input type="text" value={email} onChange={e => setEmail(e.target.value)} className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm" placeholder={"邮箱地址"} />
          </div>
          <div className="w-full md:w-64">
            <label className="text-xs font-semibold mb-1.5 block">{"密码（选填）"}</label>
            <input type="text" value={password} onChange={e => setPassword(e.target.value)} className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm" placeholder={"用于自动刷新或激活"} />
          </div>
          <Button onClick={handleAdd} variant="secondary" className="h-10 w-full md:w-auto font-semibold">
            <Plus className="mr-2 h-4 w-4" /> {"注入账号"}
          </Button>
        </div>
      </div>

      <div className="rounded-2xl border bg-card/30 overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b bg-muted/10">
          <h3 className="text-xl font-bold">{"账号列表"}</h3>
          <span className="inline-flex items-center justify-center bg-primary/10 text-primary rounded-full px-3 py-1 text-xs font-bold">{accounts.length}</span>
        </div>
        <table className="w-full text-sm text-left">
          <thead className="bg-muted/30 border-b text-muted-foreground text-xs uppercase tracking-wider font-semibold">
            <tr>
              <th className="h-12 px-6 align-middle w-12">
                <input
                  ref={pageSelectAllRef}
                  type="checkbox"
                  checked={isCurrentPageAllSelected}
                  onChange={toggleCurrentPageSelection}
                  aria-label="选择当前页账号"
                  disabled={currentPageClearableEmails.length === 0 || personalizationLoading || personalizationBusy}
                  className="h-4 w-4 rounded border-input"
                />
              </th>
              <th className="h-12 px-6 align-middle w-20">{"序号"}</th>
              <th className="h-12 px-6 align-middle">{"账号"}</th>
              <th className="h-12 px-6 align-middle">{"状态"}</th>
              <th className="h-12 px-6 align-middle">{"并发负载"}</th>
              <th className="h-12 px-6 align-middle text-right">{"操作"}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {accounts.length === 0 && (
              <tr>
                <td colSpan={6} className="px-6 py-12 text-center text-muted-foreground">{"暂无账号，请手动注入或一键获取新号。"}</td>
              </tr>
            )}
            {pagedAccounts.map((acc, index) => {
              const clearDisabled = !canClearChats(acc)
              const rowNumber = pageStartIndex + index + 1

              return (
                <tr key={acc.email} className="transition-colors hover:bg-black/5 dark:hover:bg-white/5">
                  <td className="px-6 py-4 align-middle">
                    <input
                      type="checkbox"
                      checked={selectedEmails.includes(acc.email)}
                      onChange={() => toggleSelectedEmail(acc.email)}
                      aria-label={`选择 ${acc.email}`}
                      disabled={clearDisabled || personalizationLoading || personalizationBusy}
                      className="h-4 w-4 rounded border-input"
                    />
                  </td>
                  <td className="px-6 py-4 align-middle font-mono text-muted-foreground">{rowNumber}</td>
                  <td className="px-6 py-4 align-middle font-medium font-mono text-foreground/90">{acc.email}</td>
                  <td className="px-6 py-4 align-middle">
                    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-bold ring-1 ${statusStyle(acc.status_code)}`}>
                      {statusText(acc)}
                    </span>
                  </td>
                  <td className="px-6 py-4 align-middle font-mono">
                    <span className="inline-flex items-center justify-center bg-muted/50 px-2 py-1 rounded text-xs border">
                      {acc.inflight || 0} {"线程"}
                    </span>
                  </td>
                  <td className="px-6 py-4 align-middle text-right">
                    <div className="flex items-center justify-end gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openSinglePersonalization(acc.email)}
                        disabled={personalizationBusy || personalizationLoading || clearDisabled}
                        className="text-red-600 dark:text-red-400 border-red-500/30 hover:bg-red-500/10 hover:text-red-700"
                        title={clearDisabled ? "缺少 cookies 和 token，无法设置" : "管理该账号的设置"}
                      >
                        <Trash2 className="h-4 w-4 mr-1" /> {"账号设置"}
                      </Button>
                      {acc.status_code !== "valid" && acc.status_code !== "rate_limited" && acc.status_code !== "banned" && (
                        <Button variant="outline" size="sm" onClick={() => handleActivate(acc.email)} className="text-orange-600 dark:text-orange-400 border-orange-500/30 hover:bg-orange-500/10 font-medium">
                          <MailWarning className="h-4 w-4 mr-1" /> {"激活"}
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleVerify(acc.email)}
                        disabled={verifying === acc.email}
                        title={"单独验证"}
                        aria-label={`验证账号 ${acc.email}`}
                      >
                        {verifying === acc.email ? <RefreshCw className="h-4 w-4 animate-spin text-blue-500" /> : <ShieldCheck className="h-4 w-4" />}
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDelete(acc.email)}
                        className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                        title={"删除账号"}
                        aria-label={`删除账号 ${acc.email}`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="flex flex-col gap-3 border-t bg-muted/10 px-6 py-4 text-sm text-muted-foreground md:flex-row md:items-center md:justify-between">
          <div>
            {accounts.length > 0
              ? `显示第 ${pageStartIndex + 1}-${pageEndIndex} 条，共 ${accounts.length} 条`
              : "共 0 条账号"}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2">
              <span>{"每页"}</span>
              <select
                value={pageSize}
                onChange={event => changePageSize(Number(event.target.value))}
                className="h-9 rounded-md border border-input bg-background px-2 text-sm text-foreground"
                aria-label="每页账号数量"
              >
                {PAGE_SIZE_OPTIONS.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <span>{"条"}</span>
            </label>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => goToPage(safeCurrentPage - 1)} disabled={safeCurrentPage <= 1}>
                上一页
              </Button>
              <span className="min-w-20 text-center text-foreground">
                {safeCurrentPage} / {totalPages}
              </span>
              <Button variant="outline" size="sm" onClick={() => goToPage(safeCurrentPage + 1)} disabled={safeCurrentPage >= totalPages}>
                下一页
              </Button>
            </div>
          </div>
        </div>
      </div>

      {personalizationTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4" onClick={closePersonalizationModal}>
          <div
            ref={personalizationModalRef}
            className="w-full max-w-3xl rounded-2xl border bg-background p-6 shadow-2xl max-h-[90vh] overflow-y-auto"
            onClick={e => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby={PERSONALIZATION_MODAL_TITLE_ID}
            aria-describedby={PERSONALIZATION_MODAL_DESCRIPTION_ID}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h4 id={PERSONALIZATION_MODAL_TITLE_ID} className="text-lg font-bold">
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
              <Button variant="ghost" size="sm" onClick={closePersonalizationModal} disabled={personalizationBusy} aria-label="关闭个性化设置弹窗">
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="mt-5 space-y-4">
              <section className="rounded-xl border bg-muted/20 p-4">
                <h5 className="text-sm font-semibold">{"记忆设置"}</h5>
                <div className="mt-3 space-y-3">
                  <label className="flex items-center justify-between gap-4 rounded-lg border bg-background px-3 py-2 text-sm">
                    <span>
                      <span className="font-medium">{"启用记忆"}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{"让账号保留长期记忆"}</span>
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
                      className="h-4 w-4 rounded border-input"
                    />
                  </label>
                  <label className="flex items-center justify-between gap-4 rounded-lg border bg-background px-3 py-2 text-sm">
                    <span>
                      <span className="font-medium">{"启用历史记忆"}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{"让账号保留历史上下文记忆"}</span>
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
                      className="h-4 w-4 rounded border-input"
                    />
                  </label>
                </div>
              </section>

              <section className="rounded-xl border bg-muted/20 p-4">
                <h5 className="text-sm font-semibold">{"工具设置"}</h5>
                <p className="mt-1 text-xs text-muted-foreground">{"共 9 个工具开关，保存时会按当前勾选状态同步到目标账号。"}</p>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  {PERSONALIZATION_TOOL_OPTIONS.map(option => (
                    <label key={option.key} className="flex items-center justify-between gap-4 rounded-lg border bg-background px-3 py-2 text-sm">
                      <span className="min-w-0">
                        <span className="block font-medium">{option.label}</span>
                        <span className="block text-xs text-muted-foreground">{option.description}</span>
                        <span className="block truncate text-[11px] font-mono text-muted-foreground/70">{option.key}</span>
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
                        className="h-4 w-4 rounded border-input"
                      />
                    </label>
                  ))}
                </div>
              </section>

              <section className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
                <h5 className="text-sm font-semibold text-red-700 dark:text-red-300">{"清空上游记录"}</h5>
                <p className="mt-1 text-xs text-red-700/80 dark:text-red-200/80">
                  {personalizationTarget.kind === "batch"
                    ? `如需清理所选 ${personalizationTarget.emails.length} 个账号的上游聊天记录，请输入确认短语。`
                    : `如需清理 ${personalizationTarget.email} 的上游聊天记录，请输入确认短语。`}
                </p>
                <label
                  htmlFor={PERSONALIZATION_MODAL_CONFIRM_INPUT_ID}
                  id={PERSONALIZATION_MODAL_CONFIRM_HELP_ID}
                  className="mt-3 block text-sm font-medium text-red-700 dark:text-red-300"
                >
                  {`请输入「${CLEAR_CONFIRM_TEXT}」以确认执行清理操作。`}
                </label>
                <input
                  id={PERSONALIZATION_MODAL_CONFIRM_INPUT_ID}
                  autoFocus
                  value={personalizationPhrase}
                  onChange={e => setPersonalizationPhrase(e.target.value)}
                  disabled={personalizationBusy}
                  className="mt-3 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
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
  )
}
