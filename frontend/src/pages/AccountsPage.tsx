import { useEffect, useMemo, useState } from "react"
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

type ClearTarget =
  | { kind: "batch" }
  | { kind: "single"; email: string }

const CLEAR_CONFIRM_TEXT = "清空上游记录"

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
    case "valid": return "可用"
    case "pending_activation": return "未激活"
    case "rate_limited": return "限流"
    case "banned": return "封禁"
    case "auth_error": return "认证失效"
    default: return acc.valid ? "可用" : "失效"
  }
}

function statusNote(acc: AccountItem) {
  if ((acc.rate_limited_until || 0) > Date.now() / 1000) {
    const seconds = Math.max(0, Math.ceil((acc.rate_limited_until! - Date.now() / 1000)))
    return `预计 ${seconds} 秒后恢复`
  }
  return acc.last_error || ""
}

function localizeError(error?: string) {
  if (!error) return "未知错误"
  const lower = error.toLowerCase()
  if (lower.includes("activation already in progress")) return "账号正在激活中，请稍后刷新"
  if (lower.includes("activation link or token not found")) return "激活链接或 Token 获取失败"
  if (lower.includes("token invalid") || lower.includes("token") || lower.includes("auth")) return "Token 无效或认证失败"
  return error
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
  const [clearTarget, setClearTarget] = useState<ClearTarget | null>(null)
  const [clearPhrase, setClearPhrase] = useState("")
  const [clearing, setClearing] = useState(false)

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
      .then(data => setAccounts(data.accounts || []))
      .catch(() => toast.error("刷新账号列表失败，请检查会话密钥"))
  }

  useEffect(() => {
    fetchAccounts()
  }, [])

  const stats = useMemo(() => {
    const result = { valid: 0, pending: 0, rateLimited: 0, banned: 0, invalid: 0 }
    for (const acc of accounts) {
      switch (acc.status_code) {
        case "valid": result.valid += 1; break
        case "pending_activation": result.pending += 1; break
        case "rate_limited": result.rateLimited += 1; break
        case "banned": result.banned += 1; break
        default: result.invalid += 1; break
      }
    }
    return result
  }, [accounts])

  const closeClearModal = () => {
    if (clearing) return
    setClearTarget(null)
    setClearPhrase("")
  }

  const runClearRequest = async () => {
    if (!clearTarget || clearPhrase !== CLEAR_CONFIRM_TEXT || clearing) return

    setClearing(true)
    const id = toast.loading("正在清理上游聊天记录...")

    try {
      const url =
        clearTarget.kind === "batch"
          ? `${API_BASE}/api/admin/accounts/chats`
          : `${API_BASE}/api/admin/accounts/${encodeURIComponent(clearTarget.email)}/chats`

      const res = await fetch(url, {
        method: "DELETE",
        headers: getAuthHeader(),
      })
      const { data, rawText } = await readClearResponse(res)

      if (!res.ok) {
        const errorMessage =
          (data && typeof data === "object" ? (data.error || data.reason || data.message) : null) ||
          (rawText ? rawText.trim() : "") ||
          res.statusText ||
          "请求失败"
        throw new Error(localizeError(String(errorMessage)))
      }

      if (clearTarget.kind === "batch") {
        const summary = (data && typeof data === "object" ? data.summary : null) || {}
        const message = `批量清理完成：成功 ${summary.success || 0}，失败 ${summary.failed || 0}，跳过 ${summary.skipped || 0}`
        if (data && typeof data === "object" && data.ok) {
          toast.success(message, { id, duration: 8000 })
        } else {
          toast.error(message, { id, duration: 8000 })
        }
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
      setClearTarget(null)
      setClearPhrase("")
    } catch {
      toast.error(clearTarget.kind === "batch" ? "批量清理请求失败" : "清理请求失败", { id, duration: 8000 })
      fetchAccounts()
    } finally {
      setClearing(false)
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

  const availableAccountCount = useMemo(
    () => accounts.filter(acc => Boolean(acc.cookies || acc.token)).length,
    [accounts]
  )

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
            onClick={() => setClearTarget({ kind: "batch" })}
            disabled={clearing || availableAccountCount === 0}
            className="border-red-500/30 text-red-600 hover:bg-red-500/10 hover:text-red-700 dark:text-red-400"
            title={availableAccountCount > 0 ? "清理所有可用账号的上游聊天记录" : "当前没有可清理的可用账号"}
          >
            <Trash2 className="mr-2 h-4 w-4" /> {"批量清理聊天记录"}
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
              <th className="h-12 px-6 align-middle">{"账号"}</th>
              <th className="h-12 px-6 align-middle">{"状态"}</th>
              <th className="h-12 px-6 align-middle">{"并发负载"}</th>
              <th className="h-12 px-6 align-middle">{"说明"}</th>
              <th className="h-12 px-6 align-middle text-right">{"操作"}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {accounts.length === 0 && (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center text-muted-foreground">{"暂无账号，请手动注入或一键获取新号。"}</td>
              </tr>
            )}
            {accounts.map(acc => {
              const clearDisabled = !acc.cookies && !acc.token

              return (
                <tr key={acc.email} className="transition-colors hover:bg-black/5 dark:hover:bg-white/5">
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
                  <td className="px-6 py-4 align-middle text-muted-foreground max-w-[420px] truncate" title={statusNote(acc)}>
                    {statusNote(acc) || "-"}
                  </td>
                  <td className="px-6 py-4 align-middle text-right">
                    <div className="flex items-center justify-end gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setClearTarget({ kind: "single", email: acc.email })}
                        disabled={clearing || clearDisabled}
                        className="text-red-600 dark:text-red-400 border-red-500/30 hover:bg-red-500/10 hover:text-red-700"
                        title={clearDisabled ? "缺少 cookies 和 token，无法清理" : "清理该账号的上游聊天记录"}
                      >
                        <Trash2 className="h-4 w-4 mr-1" /> {"清理聊天记录"}
                      </Button>
                      {acc.status_code !== "valid" && acc.status_code !== "rate_limited" && acc.status_code !== "banned" && (
                        <Button variant="outline" size="sm" onClick={() => handleActivate(acc.email)} className="text-orange-600 dark:text-orange-400 border-orange-500/30 hover:bg-orange-500/10 font-medium">
                          <MailWarning className="h-4 w-4 mr-1" /> {"激活"}
                        </Button>
                      )}
                      <Button variant="outline" size="sm" onClick={() => handleVerify(acc.email)} disabled={verifying === acc.email} title={"单独验证"}>
                        {verifying === acc.email ? <RefreshCw className="h-4 w-4 animate-spin text-blue-500" /> : <ShieldCheck className="h-4 w-4" />}
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => handleDelete(acc.email)} className="text-destructive hover:bg-destructive/10 hover:text-destructive" title={"删除账号"}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {clearTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4" onClick={closeClearModal}>
          <div className="w-full max-w-lg rounded-2xl border bg-background p-6 shadow-2xl" onClick={e => e.stopPropagation()}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <h4 className="text-lg font-bold">{clearTarget.kind === "batch" ? "批量清理上游聊天记录" : "清理账号上游聊天记录"}</h4>
                <p className="mt-1 text-sm text-muted-foreground">
                  {clearTarget.kind === "batch"
                    ? `将仅处理所有可用账号，当前可用账号数：${availableAccountCount}`
                    : `目标账号：${clearTarget.email}`}
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={closeClearModal} disabled={clearing}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="mt-5 rounded-xl border border-red-500/20 bg-red-500/5 p-4">
              <p className="text-sm font-medium text-red-700 dark:text-red-300">
                {`请输入「${CLEAR_CONFIRM_TEXT}」以确认执行清理操作。`}
              </p>
              <input
                autoFocus
                value={clearPhrase}
                onChange={e => setClearPhrase(e.target.value)}
                disabled={clearing}
                className="mt-3 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder={CLEAR_CONFIRM_TEXT}
              />
            </div>

            <div className="mt-6 flex justify-end gap-2">
              <Button variant="outline" onClick={closeClearModal} disabled={clearing}>
                取消
              </Button>
              <Button
                variant="destructive"
                onClick={runClearRequest}
                disabled={clearing || clearPhrase !== CLEAR_CONFIRM_TEXT}
              >
                {clearing ? "清理中..." : "确认清理"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
