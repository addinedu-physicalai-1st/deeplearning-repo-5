<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import type { EmotionLog } from '@/types'
import { getLogs } from '@/api'

const logs = ref<EmotionLog[]>([])
const page = ref(1)
const pageSize = 20
const total = ref(0)
const loading = ref(false)

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize)))

async function fetchPage(p: number) {
  loading.value = true
  try {
    const data = await getLogs(p, pageSize)
    logs.value = data.items
    total.value = data.total
    page.value = data.page
  } catch (e) {
    console.error('로그 조회 실패:', e)
  } finally {
    loading.value = false
  }
}

function prevPage() {
  if (page.value > 1) fetchPage(page.value - 1)
}

function nextPage() {
  if (page.value < totalPages.value) fetchPage(page.value + 1)
}

function emotionLabel(emotion: string): string {
  if (emotion === 'positive') return '긍정'
  if (emotion === 'negative') return '부정'
  return '없음'
}

function emotionClass(emotion: string): string {
  if (emotion === 'positive') return 'tag-green'
  if (emotion === 'negative') return 'tag-red'
  return 'tag-gray'
}

function formatTime(ts: string): string {
  return ts.replace('T', ' ').slice(0, 19)
}

onMounted(() => fetchPage(1))
</script>

<template>
  <div class="log-history">
    <div class="page-header">
      <h2 class="page-title">Emotion Log History</h2>
      <span class="total-count">{{ total }}건</span>
    </div>

    <div class="card table-wrapper">
      <div v-if="loading" class="loading-msg">로딩 중...</div>
      <table v-else class="log-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>시간</th>
            <th>객체</th>
            <th>감정</th>
            <th>상세</th>
            <th>신뢰도</th>
            <th>FPS</th>
            <th>행동지시</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="log in logs" :key="log.id">
            <td>{{ log.id }}</td>
            <td class="cell-time">{{ formatTime(log.timestamp) }}</td>
            <td>{{ log.object_type }}</td>
            <td>
              <span :class="['tag', emotionClass(log.emotion)]">
                {{ emotionLabel(log.emotion) }}
              </span>
            </td>
            <td>{{ log.emotion_detail ?? '-' }}</td>
            <td>{{ (log.emotion_conf * 100).toFixed(1) }}%</td>
            <td>{{ log.fps }}</td>
            <td class="cell-order">{{ log.order_text ?? '-' }}</td>
          </tr>
          <tr v-if="logs.length === 0">
            <td colspan="8" class="empty-row">저장된 로그가 없습니다.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="pagination">
      <button class="btn" :disabled="page <= 1" @click="prevPage">이전</button>
      <span class="page-info">{{ page }} / {{ totalPages }}</span>
      <button class="btn" :disabled="page >= totalPages" @click="nextPage">다음</button>
    </div>
  </div>
</template>

<style scoped>
.log-history {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.total-count {
  font-size: 0.85rem;
  color: var(--text-secondary);
  background: var(--bg-card);
  padding: 2px 10px;
  border-radius: 12px;
}

.table-wrapper {
  overflow-x: auto;
}

.log-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.log-table th,
.log-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

.log-table th {
  color: var(--text-secondary);
  font-weight: 500;
  font-size: 0.8rem;
  text-transform: uppercase;
}

.log-table tbody tr:hover {
  background: rgba(59, 130, 246, 0.05);
}

.cell-time {
  font-family: monospace;
  font-size: 0.8rem;
  white-space: nowrap;
}

.cell-order {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.75rem;
  font-weight: 600;
}

.tag-green {
  background: rgba(34, 197, 94, 0.15);
  color: var(--accent-green);
}

.tag-red {
  background: rgba(239, 68, 68, 0.15);
  color: var(--accent-red);
}

.tag-gray {
  background: rgba(139, 143, 163, 0.15);
  color: var(--text-secondary);
}

.empty-row {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px 0 !important;
}

.loading-msg {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px 0;
}

.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
}

.btn {
  padding: 6px 16px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-card);
  color: var(--text-primary);
  cursor: pointer;
  font-size: 0.85rem;
  transition: background 0.2s;
}

.btn:hover:not(:disabled) {
  background: var(--accent-blue);
  border-color: var(--accent-blue);
}

.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.page-info {
  font-size: 0.85rem;
  color: var(--text-secondary);
}
</style>
