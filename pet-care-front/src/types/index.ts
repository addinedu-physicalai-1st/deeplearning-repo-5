export interface PetStatus {
  fps: number
  detected: boolean
  object_type: string
  emotion: string
  emotion_detail: string | null
  emotion_conf: number
  order: string
  member_name: string
  pet_name: string
  breed: string
}

export interface LogEntry {
  type: string
  timestamp: string
  object: string
  emotion: string
  emotion_detail: string
  emotion_conf: number
  member_name: string
  pet_name: string
  breed: string
  order: string
  fps: number
}

export interface ActionGuideEntry {
  type: 'action_guide'
  emotion_trigger: string
  action: string
  control: { cmd: string; param: string } | null
}

export interface Member {
  id: number
  name: string
  email: string | null
  phone: string | null
  created_at: string
}

export interface Pet {
  id: number
  member_id: number
  name: string
  species: string
  breed: string | null
  age: number | null
  created_at: string
  member_name?: string
}

export interface EmotionLog {
  id: number
  pet_id: number | null
  timestamp: string
  object_type: string
  emotion: string
  emotion_detail: string | null
  emotion_conf: number
  fps: number
  order_text: string | null
  created_at: string
}

export interface LogPage {
  items: EmotionLog[]
  total: number
  page: number
  page_size: number
}
