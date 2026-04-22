import axiosInstance from './axiosInstance'

// Re-export axiosInstance for convenience
const axios = axiosInstance
import { getToken } from './auth'

const API_URL = '/api/monitoring/'

// 服务状态类型定义
export interface ServiceStatus {
  status: string
  details: string
  server_url?: string
  server_status?: any
}

// 服务状态监控响应类型
export interface ServiceStatusResponse {
  timestamp: string
  services: {
    local_api: ServiceStatus
    ddns_training_server: ServiceStatus
  }
}

/**
 * 获取服务状态信息
 */
export const getServiceStatus = async (): Promise<ServiceStatusResponse> => {
  const token = getToken()
  if (!token) {
    throw new Error('Not authenticated')
  }

  try {
    const response = await axios.get<ServiceStatusResponse>(`${API_URL}service-status`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    })
    return response.data
  } catch (error) {
    console.error('Failed to get service status:', error)
    throw error
  }
}

/**
 * 获取详细的服务状态信息
 */
export const getDetailedServiceStatus = async (): Promise<ServiceStatusResponse> => {
  const token = getToken()
  if (!token) {
    throw new Error('Not authenticated')
  }

  try {
    const response = await axios.get<ServiceStatusResponse>(`${API_URL}service-status/details`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    })
    return response.data
  } catch (error) {
    console.error('Failed to get detailed service status:', error)
    throw error
  }
}
