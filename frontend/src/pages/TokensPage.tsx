import { useState, useEffect } from "react"
import { Button } from "../components/ui/button"
import { Plus, RefreshCw, Copy, Check, Trash2, KeyRound } from "lucide-react"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { Card, CardContent } from "../components/ui/card"
import { Notice } from "../components/ui/field"
import { PageShell } from "../components/ui/page-shell"
import { StatusBadge } from "../components/ui/badge"

export default function TokensPage() {
  const [keys, setKeys] = useState<string[]>([])
  const [copied, setCopied] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [failed, setFailed] = useState(false)

  const fetchKeys = () => {
    fetch(`${API_BASE}/api/admin/keys`, { headers: getAuthHeader() })
      .then(res => {
        if (!res.ok) throw new Error("Unauthorized")
        return res.json()
      })
      .then(data => {
        setKeys(data.keys || [])
        setFailed(false)
      })
      .catch(() => {
        setFailed(true)
        toast.error("刷新失败，请检查会话 Key")
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    fetchKeys()
  }, [])

  const handleGenerate = () => {
    fetch(`${API_BASE}/api/admin/keys`, {
      method: "POST",
      headers: getAuthHeader()
    }).then(async res => {
      const data = await res.json().catch(() => ({}))
      if (res.ok) {
        toast.success("已生成新的 API Key")
        if (data.key) copyToClipboard(data.key)
        fetchKeys()
      } else {
        toast.error(data.detail || "生成失败，请检查权限")
      }
    }).catch(() => toast.error("生成失败，请检查权限"))
  }

  const handleDelete = (key: string) => {
    fetch(`${API_BASE}/api/admin/keys/${encodeURIComponent(key)}`, {
      method: "DELETE",
      headers: getAuthHeader()
    }).then(async res => {
      if (res.ok) {
        toast.success("API Key 已删除")
        fetchKeys()
      } else {
        const data = await res.json().catch(() => ({}))
        toast.error(data.detail || "删除失败")
      }
    }).catch(() => toast.error("删除失败"))
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(text)
    setTimeout(() => setCopied(null), 2000)
  }

  return (
    <PageShell
      actions={
        <>
          <Button variant="outline" onClick={() => { fetchKeys(); toast.success("已刷新"); }}>
            <RefreshCw className="mr-2 h-4 w-4" /> 刷新
          </Button>
          <Button onClick={handleGenerate}>
            <Plus className="mr-2 h-4 w-4" /> 生成新 Key
          </Button>
        </>
      }
    >
      <div className="space-y-6">
        {failed && <Notice tone="error">无法读取 API Key 列表，请检查会话 Key 是否正确。</Notice>}

        {/* 凭证说明 + 数量概览：先讲清这是什么，再列出条目 */}
        <Card>
          <CardContent className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                <KeyRound className="h-4 w-4 text-primary" />
              </span>
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground">下游接入凭证</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  这些 Key 用于客户端调用本网关。生成后会自动复制到剪贴板，请立即保存——列表只展示 Key 本身。
                </p>
              </div>
            </div>
            <div className="shrink-0 text-left sm:text-right">
              <p className="text-xs text-muted-foreground">当前 Key 数量</p>
              <p className="tabular text-2xl font-semibold tracking-tight text-foreground">
                {loading ? <span className="text-muted-foreground/40">—</span> : keys.length}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/40 px-5 py-4">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-foreground">凭证列表</h3>
              <StatusBadge tone="accent" className="tabular">{keys.length}</StatusBadge>
            </div>
          </div>

          <ul className="divide-y divide-border">
            {loading && keys.length === 0 && (
              <li className="px-5 py-12 text-center text-sm text-muted-foreground">正在加载…</li>
            )}
            {!loading && keys.length === 0 && (
              <li className="flex flex-col items-center gap-4 px-5 py-14 text-center">
                <KeyRound className="h-10 w-10 text-muted-foreground/25" aria-hidden="true" />
                <div>
                  <p className="text-sm font-medium text-foreground">还没有 API Key</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    点击右上角「生成新 Key」为下游客户端创建一份访问凭证。
                  </p>
                </div>
                <Button onClick={handleGenerate}>
                  <Plus className="mr-2 h-4 w-4" /> 生成新 Key
                </Button>
              </li>
            )}
            {keys.map((k, i) => (
              <li
                key={k}
                className="flex flex-col gap-3 px-5 py-3.5 transition-colors hover:bg-muted/40 lg:flex-row lg:items-center lg:gap-4"
              >
                <div className="flex min-w-0 flex-1 items-start gap-3">
                  <span className="tabular mt-0.5 shrink-0 font-mono text-xs text-muted-foreground">
                    #{i + 1}
                  </span>
                  <span className="min-w-0 break-all font-mono text-xs text-foreground">{k}</span>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Button variant="outline" size="sm" onClick={() => copyToClipboard(k)} className="gap-1.5">
                    {copied === k
                      ? <><Check className="h-3.5 w-3.5" /> 已复制</>
                      : <><Copy className="h-3.5 w-3.5" /> 复制</>}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(k)}
                    className="gap-1.5 text-destructive hover:bg-destructive/10 hover:text-destructive"
                    aria-label={`删除 API Key ${k}`}
                  >
                    <Trash2 className="h-4 w-4" /> 删除
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </PageShell>
  )
}
