<script setup lang="ts">
import { RouterLink } from 'vue-router'
</script>

<template>
  <header class="app-header">
    <div class="header-inner">
      <div class="header-brand">
        <span class="brand-icon">&#x1F43E;</span>
        <span class="brand-text">Pet Care V3</span>
      </div>
      <nav class="header-nav">
        <RouterLink to="/" class="nav-link" active-class="active">Dashboard</RouterLink>
        <RouterLink to="/logs" class="nav-link" active-class="active">Log History</RouterLink>
      </nav>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
}

.header-inner {
  max-width: 1600px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 52px;
}

.header-brand {
  display: flex;
  align-items: center;
  gap: 8px;
}

.brand-icon {
  font-size: 1.4rem;
}

.brand-text {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-primary);
}

.header-nav {
  display: flex;
  gap: 4px;
}

.nav-link {
  padding: 6px 14px;
  border-radius: 8px;
  font-size: 0.9rem;
  color: var(--text-secondary);
  transition: all 0.15s;
}

.nav-link:hover {
  background: var(--bg-card);
  color: var(--text-primary);
}

.nav-link.active {
  background: var(--accent-blue);
  color: #fff;
}
</style>
