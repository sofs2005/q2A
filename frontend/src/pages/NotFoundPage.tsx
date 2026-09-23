import { Link } from "react-router-dom"
import { Compass } from "lucide-react"
import { Button } from "../components/ui/button"
import { PageShell } from "../components/ui/page-shell"

export default function NotFoundPage() {
  return (
    <PageShell
      title="页面不存在"
      description="该地址没有对应的控制台页面，可能是链接已变更或输入有误。"
    >
      <div className="flex flex-col items-center gap-5 rounded-xl border border-border bg-card px-6 py-16 text-center">
        <Compass className="h-12 w-12 text-muted-foreground/30" aria-hidden="true" />
        <div className="space-y-1">
          <p className="text-base font-medium text-foreground">找不到这个页面</p>
          <p className="text-sm text-muted-foreground">
            请从左侧导航选择要进入的模块，或直接返回运行状态。
          </p>
        </div>
        <Button asChild>
          <Link to="/">返回运行状态</Link>
        </Button>
      </div>
    </PageShell>
  )
}
