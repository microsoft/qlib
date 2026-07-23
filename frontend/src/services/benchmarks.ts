import axiosInstance from './axiosInstance'

// Re-export axiosInstance for convenience
const axios = axiosInstance
import { getToken } from './auth'

interface Benchmark {
  id: string
  name: string
  model: string
  file_name: string
  path: string
  content: string
}

const API_URL = '/api/benchmarks/'

// 获取所有benchmark样例
export const getBenchmarks = async (): Promise<Benchmark[]> => {
  const token = getToken()
  if (!token) {
    throw new Error('Not authenticated')
  }
  
  try {
    const response = await axios.get<Benchmark[]>(API_URL, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    })
    return response.data
  } catch (error) {
    console.error('Failed to get benchmarks:', error)
    throw error
  }
}

// 获取特定benchmark样例
export const getBenchmark = async (benchmarkId: string): Promise<Benchmark> => {
  const token = getToken()
  if (!token) {
    throw new Error('Not authenticated')
  }
  
  try {
    const response = await axios.get<Benchmark>(`${API_URL}${benchmarkId}`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    })
    return response.data
  } catch (error) {
    console.error(`Failed to get benchmark ${benchmarkId}:`, error)
    throw error
  }
}
