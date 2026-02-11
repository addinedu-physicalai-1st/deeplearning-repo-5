<script setup lang="ts">
import { usePetStatusStore } from '@/stores/petStatus'

const statusStore = usePetStatusStore()
const videoUrl = '/api/video_feed'
</script>

<template>
  <div class="video-container card">
    <div class="video-wrapper">
      <img :src="videoUrl" alt="Live Feed" class="video-stream" />
      <div class="video-overlay top-left">
        <span class="live-badge">LIVE</span>
      </div>
      <div class="video-overlay top-right">
        <span class="fps-badge mono">{{ statusStore.status.fps }} FPS</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-container {
  padding: 0;
  overflow: hidden;
}

.video-wrapper {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
}

.video-stream {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.video-overlay {
  position: absolute;
  padding: 8px 12px;
}

.top-left {
  top: 0;
  left: 0;
}

.top-right {
  top: 0;
  right: 0;
}

.live-badge {
  background: var(--accent-red);
  color: #fff;
  padding: 2px 10px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.1em;
}

.fps-badge {
  background: rgba(0, 0, 0, 0.6);
  color: var(--accent-green);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
}
</style>
