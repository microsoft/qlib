import axiosInstance from './axiosInstance'

// Re-export axiosInstance for convenience
const axios = axiosInstance
import { getToken } from './auth'

const API_URL = '/api/experiments/'

interface Experiment {
  id: number
  name: string
  description: string
  config: any
  status: string
  created_at: string
  updated_at: string
  performance?: any
  error?: string
}

interface ExperimentCreate {
  name: string
  description: string
  config: any
}

// Analysis interfaces
export interface SignalAnalysis {
  ic: {
    dates: string[]
    values: number[]
  }
  monthly_ic: {
    months: string[]
    values: number[]
  }
  auto_correlation: {
    lags: number[]
    values: number[]
  }
  return_distribution: {
    bins: number[]
    counts: number[]
  }
}

export interface PortfolioAnalysis {
  cumulative_return: {
    dates: string[]
    values: number[]
  }
  group_returns: {
    dates: string[]
    groups: Record<string, number[]>
  }
  long_short: {
    dates: string[]
    values: number[]
  }
}

export interface BacktestAnalysis {
  report: {
    total_return: number
    annual_return: number
    sharpe_ratio: number
    max_drawdown: number
    win_rate: number
  }
  explanation: string
}

export interface FullAnalysis {
  signal_analysis: SignalAnalysis
  portfolio_analysis: PortfolioAnalysis
  backtest_analysis: BacktestAnalysis
}

export const getExperiments = async (): Promise<Experiment[]> => {
  const token = getToken()
  const response = await axios.get(API_URL, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getExperiment = async (id: number): Promise<Experiment> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const createExperiment = async (experiment: ExperimentCreate): Promise<Experiment> => {
  const token = getToken()
  const response = await axios.post(API_URL, experiment, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const runExperiment = async (id: number): Promise<any> => {
  const token = getToken()
  const response = await axios.post(`${API_URL}${id}/run`, {}, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const updateExperiment = async (id: number, experiment: Partial<Experiment>): Promise<Experiment> => {
  const token = getToken()
  const response = await axios.put(`${API_URL}${id}`, experiment, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const deleteExperiment = async (id: number): Promise<any> => {
  const token = getToken()
  const response = await axios.delete(`${API_URL}${id}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getProfitLoss = async (): Promise<any[]> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}profit-loss`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getExperimentLogs = async (id: number): Promise<string> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}/logs`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data.logs || ''
}

// Analysis API calls
export const getFullAnalysis = async (id: number): Promise<FullAnalysis> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}/analysis`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getSignalAnalysis = async (id: number): Promise<SignalAnalysis> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}/analysis/signal`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getPortfolioAnalysis = async (id: number): Promise<PortfolioAnalysis> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}/analysis/portfolio`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}

export const getBacktestAnalysis = async (id: number): Promise<BacktestAnalysis> => {
  const token = getToken()
  const response = await axios.get(`${API_URL}${id}/analysis/backtest`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })
  return response.data
}
