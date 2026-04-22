import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getModel } from '../services/models'

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

const ModelDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const [model, setModel] = useState<ModelVersion | null>(null)
  const [loading, setLoading] = useState(true)
  const [isMetricsExpanded, setIsMetricsExpanded] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const fetchModel = async () => {
      if (!id) return
      
      try {
        const modelData = await getModel(parseInt(id))
        setModel(modelData)
      } catch (err) {
        console.error('Failed to fetch model:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchModel()
  }, [id])

  if (loading) {
    return <div className="container">Loading...</div>
  }

  if (!model) {
    return <div className="container">Model not found</div>
  }

  return (
    <div className="container page-transition">
      <div className="page-header">
        <h1>Model: {model.name} (v{model.version})</h1>
        <button className="btn" onClick={() => navigate('/models')}>
          Back to Models
        </button>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <h3>Model Info</h3>
        <p><strong>ID:</strong> {model.id}</p>
        <p><strong>Experiment ID:</strong> {model.experiment_id}</p>
        <p><strong>Version:</strong> {model.version}</p>
        <p><strong>Path:</strong> {model.path}</p>
        <p><strong>Created At:</strong> {new Date(model.created_at).toLocaleString()}</p>
      </div>

      {model.metrics && (
        <div className="card" style={{ marginBottom: '20px' }}>
          <div className="card-header">
            <h3>Metrics</h3>
            <button 
              className="btn btn-sm"
              onClick={() => setIsMetricsExpanded(!isMetricsExpanded)}
            >
              {isMetricsExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
          {isMetricsExpanded && (
            <pre className="config-pre">
              {JSON.stringify(model.metrics, null, 2)}
            </pre>
          )}
        </div>
      )}

      {model.performance && (
        <div className="card" style={{ marginBottom: '20px' }}>
          <h3>Performance</h3>
          <p><strong>Total Return:</strong> {(model.performance.total_return * 100).toFixed(2)}%</p>
          <p><strong>Max Drawdown:</strong> {(model.performance.max_drawdown * 100).toFixed(2)}%</p>
          <p><strong>Annual Return:</strong> {(model.performance.annual_return * 100).toFixed(2)}%</p>
          <p><strong>Sharpe Ratio:</strong> {model.performance.sharpe_ratio.toFixed(2)}</p>
        </div>
      )}
    </div>
  )
}

export default ModelDetail
