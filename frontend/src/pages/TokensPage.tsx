import { useState, useEffect } from "react"
import { Button } from "../components/ui/button"
import { Plus, RefreshCw, Copy, Check, Trash2, KeyRound } from "lucide-react"
import { toast } from "sonner"
import { getAuthHeader } from "../lib/auth"
import { API_BASE } from "../lib/api"
import { Card } from "../components/ui/card"
import { Notice } from "../components/ui/field"

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
    <div className="w-full space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">API Key 分发</h2>
          <p className="mt-1 text-sm text-muted-foreground">管理可以访问此网关的下游凭证。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" onClick={() => { fetchKeys(); toast.success("已刷新"); }}>
            <RefreshCw className="mr-2 h-4 w-4" /> 刷新
          </Button>
          <Button onClick={handleGenerate}>
            <Plus className="mr-2 h-4 w-4" /> 生成新 Key
          </Button>
        </div>
      </div>

      {failed && <Notice tone="error">无法读取 API Key 列表，请检查会话 Key 是否正确。</Notice>}

      <Card className="overflow-hidden">
        {/* Desktop table */}
        <table className="hidden w-full text-left text-sm md:table">
          <thead className="border-b bg-muted/40 text-xs font-medium text-muted-foreground">
            <tr>
              <th scope="col" className="h-11 w-16 px-5 align-middle">序号</th>
              <th scope="col" className="h-11 px-5 align-middle">API Key</th>
              <th scope="col" className="h-11 px-5 text-right align-middle">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {loading && keys.length === 0 && (
              <tr>
                <td colSpan={3} className="px-5 py-12 text-center text-muted-foreground">正在加载…</td>
              </tr>
            )}
            {!loading && keys.length === 0 && (
              <tr>
                <td colSpan={3} className="px-5 py-12 text-center text-muted-foreground">暂无 API Key，点击右上角「生成新 Key」创建。</td>
              </tr>
            )}
            {keys.map((k, i) => (
              <tr key={k} className="transition-colors hover:bg-muted/40">
                <td className="tabular px-5 py-3 align-middle font-mono text-xs text-muted-foreground">{i + 1}</td>
                <td className="break-all px-5 py-3 align-middle font-mono text-xs text-foreground">{k}</td>
                <td className="px-5 py-3 align-middle">
                  <div className="flex items-center justify-end gap-2">
                    <Button variant="outline" size="sm" onClick={() => copyToClipboard(k)}>
                      {copied === k
                        ? <><Check className="mr-1.5 h-3.5 w-3.5" /> 已复制</>
                        : <><Copy className="mr-1.5 h-3.5 w-3.5" /> 复制</>}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDelete(k)}
                      className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                      aria-label={`删除 API Key ${k}`}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Mobile cards */}
        <div className="divide-y divide-border md:hidden">
          {loading && keys.length === 0 && (
            <p className="px-5 py-10 text-center text-sm text-muted-foreground">正在加载…</p>
          )}
          {!loading && keys.length === 0 && (
            <p className="px-5 py-10 text-center text-sm text-muted-foreground">暂无 API Key。</p>
          )}
          {keys.map((k, i) => (
            <div key={k} className="space-y-3 px-4 py-3.5">
              <div className="flex items-start gap-3">
                <span className="tabular mt-0.5 font-mono text-xs text-muted-foreground">#{i + 1}</span>
                <KeyRound className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                <span className="min-w-0 break-all font-mono text-xs text-foreground">{k}</span>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" className="flex-1" onClick={() => copyToClipboard(k)}>
                  {copied === k
                    ? <><Check className="mr-1.5 h-3.5 w-3.5" /> 已复制</>
                    : <><Copy className="mr-1.5 h-3.5 w-3.5" /> 复制</>}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(k)}
                  className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                  aria-label={`删除 API Key ${k}`}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
