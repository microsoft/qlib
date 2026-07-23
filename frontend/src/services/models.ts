import axiosInstance from './axiosInstance'

// Re-export axiosInstance for convenience
const axios = axiosInstance
import { getToken } from './auth'

const API_URL = '/api/models/'

interface ModelVersion {
  id: number
  name: string
  experiment_id: number
  version: number
  metrics: any
  path: string
  created_at: string
  performance?: any
}

interface ModelResponse {
  data: ModelVersion[]
  total: number
  page: number
  per_page: number
}

export const getModels = async (page: number = 1, perPage: number = 10): Promise<ModelResponse> => {
  const token = getToken()
  const response = await axios.get(API_URL, {
    params: { page, per_page: perPage },
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getModel = async (id: number): Promise<ModelVersion> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getModelVersions = async (experimentId?: number, page: number = 1, perPage: number = 10): Promise<ModelResponse> => {
  const url = experimentId ? `${API_URL}experiment/${experimentId}` : API_URL
  const token = getToken()
  const response = await axios.get(url, {
    params: { page, per_page: perPage },
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const deleteModel = async (id: number): Promise<any> => {
  const token = getToken()
  const response = await axios.delete(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}
