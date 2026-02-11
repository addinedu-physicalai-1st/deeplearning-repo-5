import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { PetStatus } from '@/types'
import { getStatus } from '@/api'

export const usePetStatusStore = defineStore('petStatus', () => {
  const status = ref<PetStatus>({
    fps: 0,
    detected: false,
    object_type: 'none',
    emotion: '대기중',
    emotion_detail: null,
    emotion_conf: 0,
    order: 'none',
    member_name: '',
    pet_name: '',
    breed: '',
  })

  let timer: ReturnType<typeof setInterval> | null = null

  async function poll() {
    try {
      status.value = await getStatus()
    } catch {
      /* 연결 실패 시 무시 */
    }
  }

  function startPolling(ms = 300) {
    if (timer) return
    poll()
    timer = setInterval(poll, ms)
  }

  function stopPolling() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  return { status, startPolling, stopPolling }
})
