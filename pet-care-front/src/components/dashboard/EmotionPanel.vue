<script setup lang="ts">
import { computed } from 'vue'
import { usePetStatusStore } from '@/stores/petStatus'

const store = usePetStatusStore()

const emotionClass = computed(() => {
  const e = store.status.emotion
  if (e === '긍정') return 'positive'
  if (e === '대기중' || e === '분석중' || e === '초기화 중...') return 'idle'
  return 'negative'
})

const confPct = computed(() => Math.round(store.status.emotion_conf * 100))
</script>

<template>
  <div class="card emotion-panel">
    <div class="card-title">Emotion</div>
    <div class="emotion-display" :class="emotionClass">
      <span class="emotion-label">{{ store.status.emotion || '---' }}</span>
      <span class="emotion-conf mono">{{ confPct }}%</span>
    </div>
    <div v-if="store.status.emotion_detail" class="emotion-detail text-muted">
      {{ store.status.emotion_detail }}
    </div>
  </div>
</template>

<style scoped>
.emotion-panel {
  text-align: center;
}

.emotion-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 16px 0;
  border-radius: 8px;
}

.emotion-display.positive {
  background: rgba(34, 197, 94, 0.1);
}

.emotion-display.negative {
  background: rgba(239, 68, 68, 0.1);
}

.emotion-display.idle {
  background: rgba(100, 116, 139, 0.1);
}

.emotion-label {
  font-size: 1.6rem;
  font-weight: 700;
}

.positive .emotion-label {
  color: var(--accent-green);
}

.negative .emotion-label {
  color: var(--accent-red);
}

.idle .emotion-label {
  color: var(--text-secondary);
}

.emotion-conf {
  font-size: 1.1rem;
  color: var(--text-secondary);
}

.emotion-detail {
  margin-top: 8px;
  font-size: 0.85rem;
}
</style>
