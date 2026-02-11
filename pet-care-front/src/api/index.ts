import type { PetStatus, Member, Pet, LogPage } from '@/types'

const BASE = '/api'

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export async function getStatus(): Promise<PetStatus> {
  return fetchJson<PetStatus>(`${BASE}/status`)
}

export async function getMembers(): Promise<Member[]> {
  return fetchJson<Member[]>(`${BASE}/members`)
}

export async function getPets(): Promise<Pet[]> {
  return fetchJson<Pet[]>(`${BASE}/pets`)
}

export async function getPetDetail(id: number) {
  return fetchJson<Pet & { recent_logs: unknown[] }>(`${BASE}/pets/${id}`)
}

export async function getLogs(page = 1, pageSize = 20): Promise<LogPage> {
  return fetchJson<LogPage>(`${BASE}/logs?page=${page}&page_size=${pageSize}`)
}

export function createLogWebSocket(
  onMessage: (data: unknown) => void,
  onClose?: () => void,
): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}${BASE}/ws/logs`)

  ws.onmessage = (evt) => {
    try {
      onMessage(JSON.parse(evt.data))
    } catch {
      /* ignore parse errors */
    }
  }
  ws.onclose = () => onClose?.()

  return ws
}
