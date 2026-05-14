import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import ts from 'typescript'

async function loadModule() {
  const source = await readFile(new URL('./registerUnlock.ts', import.meta.url), 'utf8')
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 },
  })
  const encoded = Buffer.from(outputText, 'utf8').toString('base64')
  return import(`data:text/javascript;base64,${encoded}`)
}

test('register unlock check returns false when Web Crypto digest is unavailable', async () => {
  const { checkRegisterUnlock } = await loadModule()

  const unlocked = await checkRegisterUnlock('user@example.com', 'secret', { subtle: undefined })

  assert.equal(unlocked, false)
})
