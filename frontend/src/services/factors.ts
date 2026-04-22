import axiosInstance from './axiosInstance'

// Re-export axiosInstance for convenience
const axios = axiosInstance
import { getToken } from './auth'

const API_URL = '/api/factors/'

// Factor Group interfaces
interface FactorGroup {
  id: number
  name: string
  description: string
  factor_count: number
  status: string
  created_at: string
  updated_at: string
}



// Factor interfaces
interface Factor {
  id: number
  name: string
  description: string
  formula: string
  type: string
  status: string
  group_id?: number
  group?: FactorGroup
  created_at: string
  updated_at: string
}

interface FactorCreate {
  name: string
  description: string
  formula: string
  type: string
  group_id?: number
}

// Factor Group functions
export const getFactorGroups = async (): Promise<FactorGroup[]> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}groups`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getFactorGroup = async (id: number): Promise<FactorGroup> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}groups/${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getFactorGroupWithFactors = async (id: number): Promise<FactorGroup & { factors: Factor[] }> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}groups/${id}/factors`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

// Factor functions
export const getFactors = async (): Promise<Factor[]> => {
  const token = getToken()
  const response = await axios.get(API_URL, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getFactor = async (id: number): Promise<Factor> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const createFactor = async (factor: FactorCreate): Promise<Factor> => {
  const token = getToken()
  const response = await axios.post(API_URL, factor, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const updateFactor = async (id: number, factor: Partial<Factor>): Promise<Factor> => {
  const token = getToken()
  const response = await axios.put(`${API_URL}${id}`, factor, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const deleteFactor = async (id: number): Promise<any> => {
  const token = getToken()
  const response = await axios.delete(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}
