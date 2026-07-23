import axiosInstance from './axiosInstance'

// Re-export axiosInstance for convenience
const axios = axiosInstance
import { getToken } from './auth'

const API_URL = '/api/configs/'

export type ConfigType = 'experiment_template' | 'normal'

interface Config {
  id: number
  name: string
  description: string
  content: string
  type: ConfigType
  created_at: string
  updated_at: string
}

interface ConfigCreate {
  name: string
  description: string
  content: string
  type?: ConfigType
}

export const getConfigs = async (): Promise<Config[]> => {
  const token = getToken()
  const response = await axios.get(API_URL, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getConfig = async (id: number): Promise<Config> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const createConfig = async (config: ConfigCreate): Promise<Config> => {
  const token = getToken()
  const response = await axios.post(API_URL, config, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const updateConfig = async (id: number, config: Partial<Config>): Promise<Config> => {
  const token = getToken()
  const response = await axios.put(`${API_URL}${id}`, config, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const deleteConfig = async (id: number): Promise<any> => {
  const token = getToken()
  const response = await axios.delete(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}
