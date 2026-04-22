import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getExperiment, getFullAnalysis } from '../services/experiments'
import { getModelVersions } from '../services/models'
import ReactECharts from 'echarts-for-react'
import * as yaml from 'js-yaml'
import SignalAnalysis from '../components/SignalAnalysisChart'
import PortfolioAnalysis from '../components/PortfolioAnalysisChart'
import BacktestReport from '../components/BacktestReportCard'

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

const ExperimentDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [models, setModels] = useState<ModelVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [analysisLoading, setAnalysisLoading] = useState(true)
  const [isConfigExpanded, setIsConfigExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState('signal')
  const [analysisData, setAnalysisData] = useState<any>(null)
  const navigate = useNavigate()

  useEffect(() => {
    const fetchData = async () => {
      if (!id) return
      
      try {
        setLoading(true)
        const experimentData = await getExperiment(parseInt(id))
        const modelsData = await getModelVersions(parseInt(id))
        setExperiment(experimentData)
        // Ensure modelsData is an array
        setModels(Array.isArray(modelsData) ? modelsData : [])
      } catch (err) {
        console.error('Failed to fetch experiment details:', err)
        // Set empty array on error
        setModels([])
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [id])

  // Fetch analysis data
  useEffect(() => {
    const fetchAnalysisData = async () => {
      if (!id || !experiment) return
      
      try {
        setAnalysisLoading(true)
        const data = await getFullAnalysis(parseInt(id))
        setAnalysisData(data)
      } catch (err) {
        console.error('Failed to fetch analysis data:', err)
      } finally {
        setAnalysisLoading(false)
      }
    }

    fetchAnalysisData()
  }, [id, experiment])

  if (loading) {
    return <div className="container">Loading...</div>
  }

  if (!experiment) {
    return <div className="container">Experiment not found</div>
  }

  // Prepare chart data if performance is available
  const getChartOption = () => {
    if (!experiment.performance) {
      return {
        title: {
          text: 'Performance Chart',
          left: 'center'
        },
        tooltip: {
          trigger: 'axis'
        },
        xAxis: {
          type: 'category',
          data: []
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            data: [],
            type: 'line'
          }
        ]
      }
    }

    const cumulativeReturns = experiment.performance.cumulative_returns
    const dates = Object.keys(cumulativeReturns)
    const values = Object.values(cumulativeReturns) as number[]

    return {
      title: {
        text: 'Cumulative Returns',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const date = params[0].axisValue
          const value = params[0].value
          return `${date}<br/>Cumulative Return: ${(value * 100).toFixed(2)}%`
        }
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}%'
        }
      },
      series: [
        {
          data: values.map(v => (v * 100).toFixed(2)),
          type: 'line',
          smooth: true,
          itemStyle: {
            color: '#646cff'
          }
        }
      ]
    }
  }

  return (
    <div className="container page-transition">
      <div className="page-header">
        <h1>Experiment: {experiment.name}</h1>
        <button className="btn" onClick={() => navigate('/experiments')}>
          Back to Experiments
        </button>
      </div>

      <div className="grid-layout">
        <div className="card">
          <h3>Experiment Info</h3>
          <p><strong>Status:</strong> <span className={`experiment-status status-${experiment.status}`}>{experiment.status}</span></p>
          <p><strong>Created At:</strong> {new Date(experiment.created_at).toLocaleString()}</p>
          <p><strong>Updated At:</strong> {new Date(experiment.updated_at).toLocaleString()}</p>
          <p><strong>Description:</strong> {experiment.description}</p>
          {experiment.error && (
            <div className="error-message">
              <strong>Error:</strong> {experiment.error}
            </div>
          )}
        </div>

        {experiment.performance && (
          <div className="card">
            <h3>Performance Metrics</h3>
            <p><strong>Total Return:</strong> {(experiment.performance.total_return * 100).toFixed(2)}%</p>
            <p><strong>Max Drawdown:</strong> {(experiment.performance.max_drawdown * 100).toFixed(2)}%</p>
            <p><strong>Annual Return:</strong> {(experiment.performance.annual_return * 100).toFixed(2)}%</p>
            <p><strong>Sharpe Ratio:</strong> {experiment.performance.sharpe_ratio.toFixed(2)}</p>
          </div>
        )}
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <h3>YAML Config</h3>
          <button 
            className="btn btn-sm"
            onClick={() => setIsConfigExpanded(!isConfigExpanded)}
          >
            {isConfigExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
        {isConfigExpanded && (
          <pre className="config-pre">
            {yaml.dump(experiment.config, { indent: 2 })}
          </pre>
        )}
      </div>

      {experiment.performance && (
        <div className="chart-container">
          <h2 className="chart-title">Performance Chart</h2>
          <ReactECharts option={getChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
      )}

      {/* Graphical Reports Section */}
      {experiment.status === 'completed' && (
        <div className="card">
          <div className="card-header">
            <h2>Graphical Reports Analysis</h2>
            <div className="report-tabs">
              <button 
                className={`tab-btn ${activeTab === 'signal' ? 'active' : ''}`}
                onClick={() => setActiveTab('signal')}
              >
                Forecasting Signal Analysis
              </button>
              <button 
                className={`tab-btn ${activeTab === 'portfolio' ? 'active' : ''}`}
                onClick={() => setActiveTab('portfolio')}
              >
                Portfolio Analysis
              </button>
              <button 
                className={`tab-btn ${activeTab === 'backtest' ? 'active' : ''}`}
                onClick={() => setActiveTab('backtest')}
              >
                Backtest Return Analysis
              </button>
            </div>
          </div>
          
          {analysisLoading ? (
            <div className="loading-container">Loading analysis data...</div>
          ) : analysisData ? (
            <div className="report-content">
              {activeTab === 'signal' && (
                <SignalAnalysis data={analysisData.signal_analysis} />
              )}
              {activeTab === 'portfolio' && (
                <PortfolioAnalysis data={analysisData.portfolio_analysis} />
              )}
              {activeTab === 'backtest' && (
                <BacktestReport data={analysisData.backtest_analysis} />
              )}
            </div>
          ) : (
            <div className="error-container">Failed to load analysis data</div>
          )}
        </div>
      )}

      <div className="section">
        <h2>Model Versions</h2>
        {models.length > 0 ? (
          <div className="models-list">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Version</th>
                  <th>Created At</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map(model => (
                  <tr key={model.id}>
                    <td>{model.name}</td>
                    <td>{model.version}</td>
                    <td>{new Date(model.created_at).toLocaleString()}</td>
                    <td>
                      <button 
                        className="btn btn-sm"
                        onClick={() => navigate(`/models/${model.id}`)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No models available for this experiment</p>
        )}
      </div>
    </div>
  )
}

export default ExperimentDetail
