import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { Member, Pet } from '@/types'
import { getMembers, getPets } from '@/api'

export const useMemberStore = defineStore('memberStore', () => {
  const members = ref<Member[]>([])
  const pets = ref<Pet[]>([])
  const loaded = ref(false)

  async function load() {
    if (loaded.value) return
    const [m, p] = await Promise.all([getMembers(), getPets()])
    members.value = m
    pets.value = p
    loaded.value = true
  }

  return { members, pets, loaded, load }
})
