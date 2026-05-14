const REGISTER_UNLOCK_HASH = "29bb93e7473e47595a454ea0c7996f659035bc5298faf820039fbf7641906aea"

function bytesToHex(buffer: ArrayBuffer) {
  return Array.from(new Uint8Array(buffer)).map(byte => byte.toString(16).padStart(2, "0")).join("")
}

export async function checkRegisterUnlock(email: string, password: string, cryptoProvider: Crypto | undefined = globalThis.crypto) {
  if (!email || !password || !cryptoProvider?.subtle) return false

  try {
    const buffer = await cryptoProvider.subtle.digest("SHA-256", new TextEncoder().encode(`${email}::${password}`))
    return bytesToHex(buffer) === REGISTER_UNLOCK_HASH
  } catch {
    return false
  }
}
