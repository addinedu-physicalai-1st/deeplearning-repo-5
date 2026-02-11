import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { LogEntry, ActionGuideEntry } from '@/types'
import { createLogWebSocket } from '@/api'

export const useLogStore = defineStore('logStore', () => {
  const logs = ref<LogEntry[]>([])
  const latestAction = ref<string>('none')
  let ws: WebSocket | null = null

  function connect() {
    if (ws && ws.readyState <= 1) return

    ws = createLogWebSocket(
      (data) => {
        const d = data as Record<string, unknown>
        if (d.type === 'log') {
          logs.value.unshift(d as unknown as LogEntry)
          if (logs.value.length > 50) logs.value.pop()
          // 가이드 없는 로그 수신 시 화면의 행동지시 초기화
          if (!d.order || d.order === 'none') {
            latestAction.value = 'none'
          }
        } else if (d.type === 'action_guide') {
          const ag = d as unknown as ActionGuideEntry
          latestAction.value = ag.action || 'none'
        }
      },
      () => {
        setTimeout(connect, 3000)
      },
    )
  }

  function disconnect() {
    ws?.close()
    ws = null
  }

  return { logs, latestAction, connect, disconnect }
})
