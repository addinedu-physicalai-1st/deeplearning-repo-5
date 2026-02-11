<script setup lang="ts">
import { usePetStatusStore } from '@/stores/petStatus'

const store = usePetStatusStore()
</script>

<template>
  <div class="card pet-info">
    <div class="card-title">Active Pet</div>
    <div class="info-grid">
      <div class="info-row">
        <span class="info-label">Owner</span>
        <span class="info-value">{{ store.status.member_name || '---' }}</span>
      </div>
      <div class="info-row">
        <span class="info-label">Pet Name</span>
        <span class="info-value">{{ store.status.pet_name || '---' }}</span>
      </div>
      <div class="info-row">
        <span class="info-label">Breed</span>
        <span class="info-value">{{ store.status.breed || '---' }}</span>
      </div>
      <div class="info-row">
        <span class="info-label">Detected</span>
        <span class="info-value">
          <span v-if="store.status.detected" class="badge badge-green">
            {{ store.status.object_type === 'dog' ? 'Dog' : 'Cat' }}
          </span>
          <span v-else class="badge badge-yellow">None</span>
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.info-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.info-label {
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.info-value {
  font-weight: 600;
  font-size: 0.9rem;
}
</style>
