<script setup lang="ts">
import { computed } from 'vue'
import { useLogStore } from '@/stores/logStore'
import { usePetStatusStore } from '@/stores/petStatus'

const logStore = useLogStore()
const statusStore = usePetStatusStore()

const order = computed(() => {
  const ws = logStore.latestAction
  const poll = statusStore.status.order
  if (ws && ws !== 'none') return ws
  if (poll && poll !== 'none') return poll
  return '대기중'
})

const isActive = computed(() => order.value !== '대기중' && order.value !== 'none')
</script>

<template>
  <div class="card action-guide">
    <div class="card-title">Action Guide</div>
    <div class="guide-content" :class="{ active: isActive }">
      <div class="guide-icon">{{ isActive ? '&#x26A0;' : '&#x2705;' }}</div>
      <p class="guide-text">{{ order }}</p>
    </div>
  </div>
</template>

<style scoped>
.guide-content {
  padding: 12px;
  border-radius: 8px;
  background: rgba(100, 116, 139, 0.1);
  text-align: center;
  transition: background 0.3s;
}

.guide-content.active {
  background: rgba(234, 179, 8, 0.1);
  border: 1px solid rgba(234, 179, 8, 0.3);
}

.guide-icon {
  font-size: 1.5rem;
  margin-bottom: 6px;
}

.guide-text {
  font-size: 0.95rem;
  color: var(--text-primary);
  line-height: 1.5;
}
</style>
