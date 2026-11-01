import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

const root = new URL("../src/", import.meta.url)

function read(path) {
  return readFileSync(new URL(path, root), "utf8")
}

test("media library page is wired into the router", () => {
  const app = read("App.tsx")
  assert.match(app, /import MediaLibraryPage from "\.\/pages\/MediaLibraryPage"/)
  assert.match(app, /<Route path="media" element=\{<MediaLibraryPage \/>\} \/>/)
})

test("media library is reachable from the sidebar", () => {
  const nav = read("layouts/navigation.ts")
  assert.match(nav, /name: "媒体库"/)
  assert.match(nav, /path: "\/media"/)
})

test("lightbox is a keyboard-operable modal", () => {
  const source = read("components/ui/lightbox.tsx")
  for (const needle of ['"Escape"', '"ArrowLeft"', '"ArrowRight"', 'role="dialog"', 'aria-modal="true"']) {
    assert.ok(source.includes(needle), `expected the lightbox to handle ${needle}`)
  }
  // 打开时锁滚动，关闭/卸载时还原，避免遮罩关掉后页面仍然滚不动
  assert.match(source, /document\.body\.style\.overflow = "hidden"/)
})

test("image and video pages reuse the shared lightbox", () => {
  for (const page of ["pages/ImagePage.tsx", "pages/VideoPage.tsx"]) {
    const source = read(page)
    assert.match(
      source,
      /import \{ Lightbox \} from "\.\.\/components\/ui\/lightbox"/,
      `expected ${page} to import the shared lightbox`,
    )
    assert.match(source, /<Lightbox/, `expected ${page} to render the lightbox`)
    assert.match(source, /previewIndex/, `expected ${page} to track the preview index`)
  }
})

test("media library page lists and deletes through the admin API", () => {
  const source = read("pages/MediaLibraryPage.tsx")
  assert.match(source, /<PageShell/)
  assert.ok(source.includes("api/admin/media"), "expected the media admin endpoints")
  assert.ok(source.includes('"DELETE"'), "expected a DELETE request for single-item removal")
  assert.ok(source.includes("api/admin/media/delete"), "expected the batch delete endpoint")
  // 删除必须走自定义确认弹窗，而不是阻塞式的 window.confirm
  assert.doesNotMatch(source, /window\.confirm/)
  assert.match(source, /role="dialog"/)
})
