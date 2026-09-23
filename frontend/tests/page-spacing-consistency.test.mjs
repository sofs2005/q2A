import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

const root = new URL("../src/", import.meta.url)

function read(path) {
  return readFileSync(new URL(path, root), "utf8")
}

/** 页面根块必须是纯布局容器：只有间距，不设宽度。 */
function pageRootClass(source, page) {
  const match = source.match(/<PageShell[\s\S]*?>\s*<div className="([^"]+)"/)
  assert.ok(match, `expected ${page} to open a root content div inside PageShell`)
  return match[1]
}

test("pages share the same content rhythm", () => {
  const pages = [
    "pages/Dashboard.tsx",
    "pages/AccountsPage.tsx",
    "pages/TokensPage.tsx",
    "pages/SettingsPage.tsx",
    "pages/ImagePage.tsx",
    "pages/TestPage.tsx",
    "pages/VideoPage.tsx",
  ]

  for (const page of pages) {
    const rootClasses = pageRootClass(read(page), page)
    assert.match(rootClasses, /(?:^|\s)space-y-\d+(?:\s|$)/, `${page} should space its sections`)
    assert.doesNotMatch(rootClasses, /max-w-/, `${page} should not set its own width`)
  }
})

test("page header band stays sticky and hosts the page description", () => {
  const source = read("components/ui/page-header.tsx")
  assert.match(source, /export function PageHeaderBand/)
  assert.match(source, /sticky top-0/)
  // 面包屑提供“当前位置”，避免依赖浏览器后退按钮找路。
  assert.match(source, /aria-label="当前位置"/)
  assert.match(source, /aria-current="page"/)
})

test("settings usage example still scrolls horizontally", () => {
  const source = read("pages/SettingsPage.tsx")
  assert.match(source, /overflow-x-auto/)
})
