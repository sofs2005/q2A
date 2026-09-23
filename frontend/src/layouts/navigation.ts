import {
  Activity,
  Film,
  Image as ImageIcon,
  KeyRound,
  LayoutDashboard,
  MessageSquare,
  Settings,
  type LucideIcon,
} from "lucide-react"

/** 侧栏分组顺序即产品心智顺序：先看运行状态，再管上游与凭证，最后是工具与配置。 */
export const NAV_GROUPS: {
  label: string
  items: { name: string; path: string; icon: LucideIcon; description: string }[]
}[] = [
  {
    label: "概览",
    items: [
      {
        name: "运行状态",
        path: "/",
        icon: LayoutDashboard,
        description: "账号池健康与网关接口总览",
      },
    ],
  },
  {
    label: "上游资源",
    items: [
      {
        name: "账号池",
        path: "/accounts",
        icon: Activity,
        description: "上游账号的注入、巡检与状态管理",
      },
    ],
  },
  {
    label: "接入凭证",
    items: [
      {
        name: "API Key",
        path: "/tokens",
        icon: KeyRound,
        description: "发放给下游客户端的访问凭证",
      },
    ],
  },
  {
    label: "生成工作台",
    items: [
      {
        name: "接口测试",
        path: "/test",
        icon: MessageSquare,
        description: "用对话验证分发链路是否正常",
      },
      {
        name: "图片生成",
        path: "/images",
        icon: ImageIcon,
        description: "按提示词与比例生成图片",
      },
      {
        name: "视频生成",
        path: "/videos",
        icon: Film,
        description: "按提示词、比例与时长生成视频",
      },
    ],
  },
  {
    label: "系统配置",
    items: [
      {
        name: "系统设置",
        path: "/settings",
        icon: Settings,
        description: "控制台认证、网关运行与模型路由",
      },
    ],
  },
]

export type RouteMeta = { group: string; name: string; description: string }

const ROUTE_META: Record<string, RouteMeta> = Object.fromEntries(
  NAV_GROUPS.flatMap(group =>
    group.items.map(item => [item.path, { group: group.label, name: item.name, description: item.description }]),
  ),
)

/** 未知路径返回 undefined，由调用方决定兜底文案。 */
export function routeMeta(pathname: string): RouteMeta | undefined {
  return ROUTE_META[pathname]
}
