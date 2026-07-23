import axiosInstance from './axiosInstance'

// Re-export axiosInstance for convenience
const axios = axiosInstance
import { getToken } from './auth'

const API_URL = '/api/data/'

interface StockData {
  id: number
  stock_code: string
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  created_at: string
  updated_at: string
  [key: string]: string | number // For custom features
}

interface DataFilter {
  stock_code?: string
  market?: string
  start_date?: string
  end_date?: string
  page?: number
  per_page?: number
  features?: string[]
  name_filter?: string
  expression_filter?: string
}

interface DataResponse {
  data: StockData[]
  total: number
  page: number
  per_page: number
}

interface InstrumentFilter {
  market?: string
  name_filter?: string
  expression_filter?: string
}

interface CalendarResponse {
  dates: string[]
  start_date: string
  end_date: string
  freq: string
}

interface FeatureExpression {
  expression: string
  alias?: string
}

interface CustomFeatureRequest {
  instruments: string[]
  features: FeatureExpression[]
  start_date: string
  end_date: string
  freq?: string
}

interface CustomFeatureResponse {
  data: Record<string, Record<string, number>>
  features: string[]
}

export const getStockData = async (filter: DataFilter): Promise<DataResponse> => {
  const token = getToken()
  const response = await axios.get(API_URL, {
    params: filter,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getStockCodes = async (): Promise<string[]> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}stock-codes`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getDataById = async (id: number): Promise<StockData> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

// New API methods
export const getTradingCalendar = async (start_date: string, end_date: string, freq: string = 'day'): Promise<CalendarResponse> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}calendar`, {
    params: { start_date, end_date, freq },
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getInstruments = async (filter: InstrumentFilter = {}): Promise<string[]> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}instruments`, {
    params: filter,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const calculateCustomFeatures = async (request: CustomFeatureRequest): Promise<CustomFeatureResponse> => {
  const token = getToken()
  const response = await axios.post(`${API_URL}features`, request, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

interface AlignDataParams {
  mode: 'auto' | 'manual'
  date: string
}

interface AlignDataResponse {
  message: string
  result: Record<string, string | number | Record<string, unknown>>
}

export const alignData = async (params: AlignDataParams): Promise<AlignDataResponse> => {
  const token = getToken()
  const response = await axios.post(`${API_URL}align`, params, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}
