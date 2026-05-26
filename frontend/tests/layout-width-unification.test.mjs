import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

const root = new URL("../src/", import.meta.url)

function read(path) {
  return readFileSync(new URL(path, root), "utf8")
}

test("admin layout owns the unified content width", () => {
  const source = read("layouts/AdminLayout.tsx")
  assert.match(source, /max-w-7xl/)
})

test("pages no longer set their own outer max width", () => {
  const pages = [
    "pages/Dashboard.tsx",
    "pages/TokensPage.tsx",
    "pages/SettingsPage.tsx",
    "pages/ImagePage.tsx",
    "pages/TestPage.tsx",
  ]

  for (const page of pages) {
    const source = read(page)
    const rootMatch = source.match(/return \([\s\S]*?<div className="([^"]+)"/)
    assert.ok(rootMatch, `expected a root className in ${page}`)
    assert.doesNotMatch(rootMatch[1], /max-w-/)
    assert.match(rootMatch[1], /w-full/)
  }
})
