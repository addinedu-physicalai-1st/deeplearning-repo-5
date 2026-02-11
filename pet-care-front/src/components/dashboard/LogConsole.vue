<script setup lang="ts">
import { useLogStore } from '@/stores/logStore'

const store = useLogStore()

function emotionBadge(e: string) {
  if (e === 'positive') return 'badge-green'
  if (e === 'negative') return 'badge-red'
  return 'badge-yellow'
}
</script>

<template>
  <div class="card log-console">
    <div class="card-title">Log Console (5s interval)</div>
    <div class="log-scroll">
      <div v-if="store.logs.length === 0" class="log-empty text-muted">Waiting for logs...</div>
      <div v-for="(log, i) in store.logs" :key="i" class="log-entry">
        <span class="log-time mono">{{ log.timestamp?.split(' ')[1] || '' }}</span>
        <span class="log-obj">{{ log.object || 'none' }}</span>
        <span class="badge" :class="emotionBadge(log.emotion)">{{ log.emotion_detail || log.emotion }}</span>
        <span class="log-conf mono">{{ Math.round(log.emotion_conf * 100) }}%</span>
        <span v-if="log.order && log.order !== 'none'" class="log-order text-yellow">{{ log.order }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.log-console {
  display: flex;
  flex-direction: column;
  max-height: 360px;
}

.log-scroll {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.log-empty {
  padding: 20px;
  text-align: center;
  font-size: 0.85rem;
}

.log-entry {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  background: var(--bg-secondary);
  font-size: 0.8rem;
}

.log-time {
  color: var(--text-secondary);
  flex-shrink: 0;
}

.log-obj {
  color: var(--accent-purple);
  font-weight: 600;
  flex-shrink: 0;
  min-width: 32px;
}

.log-conf {
  color: var(--text-secondary);
  flex-shrink: 0;
}

.log-order {
  font-size: 0.75rem;
  margin-left: auto;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
