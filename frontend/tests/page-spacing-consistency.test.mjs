import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

const root = new URL("../src/", import.meta.url)

function read(path) {
  return readFileSync(new URL(path, root), "utf8")
}

function rootClass(source) {
  const match = source.match(/return \([\s\S]*?<div className="([^"]+)"/)
  assert.ok(match, "expected a root className")
  return match[1]
}

function topBarClass(source) {
  const match = source.match(/<div className="([^"]*justify-between[^"]*)">\s*<div>/)
  assert.ok(match, "expected a top bar className")
  return match[1]
}

test("Tokens, Settings, and Test pages share the same page rhythm", () => {
  const pages = [
    ["pages/TokensPage.tsx", true],
    ["pages/SettingsPage.tsx", true],
    ["pages/TestPage.tsx", false],
  ]

  for (const [path] of pages) {
    const source = read(path)
    const rootClasses = rootClass(source)
    assert.match(rootClasses, /(?:^|\s)w-full(?:\s|$)/)
    assert.match(rootClasses, /(?:^|\s)space-y-6(?:\s|$)/)
    assert.doesNotMatch(rootClasses, /max-w-/)

    const barClasses = topBarClass(source)
    assert.match(barClasses, /flex/)
    assert.match(barClasses, /gap-4/)
    assert.match(barClasses, /md:flex-row/)
    assert.match(barClasses, /md:items-start/)
  }
})

test("Settings usage example still scrolls horizontally", () => {
  const source = read("pages/SettingsPage.tsx")
  assert.match(source, /overflow-x-auto/)
})
