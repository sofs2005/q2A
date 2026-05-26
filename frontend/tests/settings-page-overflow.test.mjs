import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

const source = readFileSync(new URL("../src/pages/SettingsPage.tsx", import.meta.url), "utf8")

test("usage example code block cannot expand the settings page width", () => {
  const match = source.match(/<div className="([^"]*overflow-x-auto[^"]*)">\s*\{curlExample\}/)

  assert.ok(match, "expected the usage example code block to keep horizontal overflow inside itself")

  const className = match[1]
  for (const requiredClass of ["min-w-0", "max-w-full", "overflow-x-auto", "whitespace-pre"]) {
    assert.match(className, new RegExp(`(?:^|\\s)${requiredClass}(?:\\s|$)`))
  }
})
