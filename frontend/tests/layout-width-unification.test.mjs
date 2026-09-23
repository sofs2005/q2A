import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

const root = new URL("../src/", import.meta.url)

function read(path) {
  return readFileSync(new URL(path, root), "utf8")
}

const PAGES = [
  "pages/Dashboard.tsx",
  "pages/AccountsPage.tsx",
  "pages/TokensPage.tsx",
  "pages/SettingsPage.tsx",
  "pages/ImagePage.tsx",
  "pages/TestPage.tsx",
  "pages/VideoPage.tsx",
  "pages/NotFoundPage.tsx",
]

test("page shell owns the unified content width", () => {
  const source = read("components/ui/page-shell.tsx")
  assert.match(source, /max-w-7xl/)
})

test("every managed page renders through the shared page shell", () => {
  for (const page of PAGES) {
    const source = read(page)
    assert.match(source, /<PageShell/, `expected ${page} to render inside PageShell`)
  }
})

test("pages no longer set their own outer max width", () => {
  for (const page of PAGES) {
    const source = read(page)
    // 只有页面根内容块受约束；内部的弹窗、气泡等局部宽度是合理的。
    const match = source.match(/<PageShell[\s\S]*?>\s*<div className="([^"]+)"/)
    assert.ok(match, `expected ${page} to open a root content div inside PageShell`)
    assert.doesNotMatch(
      match[1],
      /\bmax-w-/,
      `${page} should let PageShell own the outer width`,
    )
  }
})

test("page shell keeps one canonical content gutter", () => {
  const source = read("components/ui/page-shell.tsx")
  assert.match(source, /mx-auto w-full max-w-7xl/)
  assert.match(source, /px-4/)
  assert.match(source, /sm:px-6/)
})
