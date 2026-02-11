<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'
import { usePetStatusStore } from '@/stores/petStatus'
import { useLogStore } from '@/stores/logStore'
import { useMemberStore } from '@/stores/memberStore'
import VideoFeed from '@/components/dashboard/VideoFeed.vue'
import EmotionPanel from '@/components/dashboard/EmotionPanel.vue'
import ActionGuide from '@/components/dashboard/ActionGuide.vue'
import LogConsole from '@/components/dashboard/LogConsole.vue'
import PetInfoCard from '@/components/dashboard/PetInfoCard.vue'

const statusStore = usePetStatusStore()
const logStore = useLogStore()
const memberStore = useMemberStore()

onMounted(() => {
  statusStore.startPolling()
  logStore.connect()
  memberStore.load()
})

onUnmounted(() => {
  statusStore.stopPolling()
  logStore.disconnect()
})
</script>

<template>
  <div class="dashboard">
    <!-- 상단: 영상 + 상태 패널 -->
    <div class="dashboard-top">
      <div class="video-section">
        <VideoFeed />
      </div>
      <div class="side-panels">
        <PetInfoCard />
        <EmotionPanel />
        <ActionGuide />
      </div>
    </div>
    <!-- 하단: 로그 콘솔 -->
    <div class="dashboard-bottom">
      <LogConsole />
    </div>
  </div>
</template>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.dashboard-top {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 16px;
}

.video-section {
  min-height: 400px;
}

.side-panels {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.dashboard-bottom {
  max-height: 320px;
}

@media (max-width: 1024px) {
  .dashboard-top {
    grid-template-columns: 1fr;
  }
}
</style>
