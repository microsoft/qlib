import React, { useEffect, useState } from 'react'
import { getExperiments } from '../services/experiments'
import { getModels } from '../services/models'
import ReactECharts from 'echarts-for-react'
import { Link } from 'react-router-dom'

interface Experiment {
  id: number
  name: string
  status: string
  created_at: string
}

interface Model {
  id: number
  name: string
  experiment_id: number
  created_at: string
}

interface PerformanceMetrics {
  total_return: number
  annual_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  avg_profit: number
  avg_loss: number
}

interface RiskMetrics {
  value_at_risk: number
  beta: number
  alpha: number
  volatility: number
}

const Dashboard: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [performanceMetrics, _setPerformanceMetrics] = useState<PerformanceMetrics>({
    total_return: 0.156,
    annual_return: 0.089,
    sharpe_ratio: 1.85,
    max_drawdown: -0.123,
    win_rate: 0.62,
    total_trades: 145,
    avg_profit: 0.025,
    avg_loss: -0.018
  })
  const [riskMetrics, _setRiskMetrics] = useState<RiskMetrics>({
    value_at_risk: 4.2,
    beta: 0.85,
    alpha: 0.023,
    volatility: 0.156
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setError(null)
        // 尝试从localStorage获取缓存数据
        const cachedExperiments = localStorage.getItem('dashboard_experiments')
        const cachedModels = localStorage.getItem('dashboard_models')
        const cacheTime = localStorage.getItem('dashboard_cache_time')
        const now = Date.now()
        const CACHE_DURATION = 5 * 60 * 1000 // 5分钟缓存

        if (cachedExperiments && cachedModels && cacheTime && (now - parseInt(cacheTime) < CACHE_DURATION)) {
          // 使用缓存数据
          setExperiments(JSON.parse(cachedExperiments))
          setModels(JSON.parse(cachedModels))
        } else {
          // 从API获取最新数据
          const experimentsData = await getExperiments()
          const modelsData = await getModels()
          // Ensure experimentsData is an array
          const safeExperiments = Array.isArray(experimentsData) ? experimentsData : []
          const safeModels = Array.isArray(modelsData) ? modelsData : []
          setExperiments(safeExperiments)
          setModels(safeModels)
          // 缓存数据
          localStorage.setItem('dashboard_experiments', JSON.stringify(safeExperiments))
          localStorage.setItem('dashboard_models', JSON.stringify(safeModels))
          localStorage.setItem('dashboard_cache_time', now.toString())
        }
      } catch (err) {
        console.error('Failed to fetch data:', err)
        // Set empty arrays on error
        setExperiments([])
        setModels([])
        setError('Failed to load dashboard data. Please try again later.')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  // Ensure experiments is an array before calling filter
  const safeExperiments = Array.isArray(experiments) ? experiments : []
  const runningExperiments = safeExperiments.filter(exp => exp.status === 'running')
  const completedExperiments = safeExperiments.filter(exp => exp.status === 'completed')
  const failedExperiments = safeExperiments.filter(exp => exp.status === 'failed')
  const createdExperiments = safeExperiments.filter(exp => exp.status === 'created')

  // Chart options for experiment status distribution
  const pieChartOptions = {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e0e0e0',
      borderWidth: 1,
      borderRadius: 8,
      textStyle: {
        color: '#333'
      }
    },
    legend: {
            orient: 'horizontal',
            bottom: 10,
            data: ['已创建', '运行中', '已完成', '失败'],
            textStyle: {
              color: '#666',
              fontSize: 12
            },
            itemWidth: 10,
            itemHeight: 10,
            itemGap: 20
          },
    series: [
      {
        name: '实验状态',
        type: 'pie',
        radius: ['45%', '70%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 12,
          borderColor: '#fff',
          borderWidth: 3
        },
        label: {
          show: false,
          position: 'center'
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 24,
            fontWeight: 'bold',
            color: '#333'
          },
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.3)'
          }
        },
        labelLine: {
          show: false
        },
        data: [
          {value: createdExperiments.length,
            name: '已创建',
            itemStyle: { color: '#64b5f6' }
          },
          {
            value: runningExperiments.length,
            name: '运行中',
            itemStyle: { color: '#81c784' }
          },
          {
            value: completedExperiments.length,
            name: '已完成',
            itemStyle: { color: '#9575cd' }
          },
          {
            value: failedExperiments.length,
            name: '失败',
            itemStyle: { color: '#ef5350' }
          }
        ]
      }
    ]
  }

  if (loading) {
    return (
      <div className="container">
        <div className="loading-container">
          <div className="loading">加载仪表盘</div>
        </div>
      </div>
    )
  }

  return (
    <div className="container page-transition">
      <div className="dashboard-header">
        <h1>仪表盘</h1>
        <p className="dashboard-subtitle">欢迎使用QLib管理系统</p>
      </div>
      
      {error && (
        <div className="error-message">
          {error}
          <button className="btn btn-sm btn-secondary" onClick={() => window.location.reload()}>
            重试
          </button>
        </div>
      )}
      
      <div className="dashboard-stats">
        <div className="stat-card">
          <div className="stat-icon total-experiments">📊</div>
          <div className="stat-content">
            <h3 className="stat-title">总实验数</h3>
            <p className="stat-value">{experiments.length}</p>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon running-experiments">🚀</div>
          <div className="stat-content">
            <h3 className="stat-title">运行中实验</h3>
            <p className="stat-value">{runningExperiments.length}</p>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon completed-experiments">✅</div>
          <div className="stat-content">
            <h3 className="stat-title">已完成实验</h3>
            <p className="stat-value">{completedExperiments.length}</p>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon failed-experiments">❌</div>
          <div className="stat-content">
            <h3 className="stat-title">失败实验</h3>
            <p className="stat-value">{failedExperiments.length}</p>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon total-models">🤖</div>
          <div className="stat-content">
            <h3 className="stat-title">总模型数</h3>
            <p className="stat-value">{models.length}</p>
          </div>
        </div>
      </div>
      
      {/* 绩效指标卡片 */}
      <div className="dashboard-metrics">
        <div className="metrics-section">
          <h2 className="metrics-section-title">核心绩效指标</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">总收益</h3>
                <span className="metric-icon">💰</span>
              </div>
              <div className={`metric-value ${performanceMetrics.total_return >= 0 ? 'positive' : 'negative'}`}>
                {(performanceMetrics.total_return * 100).toFixed(2)}%
              </div>
              <div className="metric-description">所有交易的总收益率</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">年化收益</h3>
                <span className="metric-icon">📈</span>
              </div>
              <div className={`metric-value ${performanceMetrics.annual_return >= 0 ? 'positive' : 'negative'}`}>
                {(performanceMetrics.annual_return * 100).toFixed(2)}%
              </div>
              <div className="metric-description">年化收益率</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">夏普比率</h3>
                <span className="metric-icon">📊</span>
              </div>
              <div className={`metric-value ${performanceMetrics.sharpe_ratio >= 1 ? 'positive' : 'warning'}`}>
                {performanceMetrics.sharpe_ratio.toFixed(2)}
              </div>
              <div className="metric-description">风险调整后收益</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">最大回撤</h3>
                <span className="metric-icon">📉</span>
              </div>
              <div className={`metric-value ${performanceMetrics.max_drawdown > -0.2 ? 'positive' : 'negative'}`}>
                {(performanceMetrics.max_drawdown * 100).toFixed(2)}%
              </div>
              <div className="metric-description">最大资金回撤</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">胜率</h3>
                <span className="metric-icon">🎯</span>
              </div>
              <div className={`metric-value ${performanceMetrics.win_rate >= 0.5 ? 'positive' : 'negative'}`}>
                {(performanceMetrics.win_rate * 100).toFixed(2)}%
              </div>
              <div className="metric-description">盈利交易占比</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">总交易次数</h3>
                <span className="metric-icon">🔄</span>
              </div>
              <div className="metric-value">{performanceMetrics.total_trades}</div>
              <div className="metric-description">所有交易总次数</div>
            </div>
          </div>
        </div>
        
        {/* 风险指标卡片 */}
        <div className="metrics-section">
          <h2 className="metrics-section-title">风险指标</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">风险价值 (VaR)</h3>
                <span className="metric-icon">⚠️</span>
              </div>
              <div className={`metric-value ${riskMetrics.value_at_risk < 0.05 ? 'positive' : 'warning'}`}>
                {(riskMetrics.value_at_risk * 100).toFixed(2)}%
              </div>
              <div className="metric-description">置信区间内最大可能损失</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">贝塔系数</h3>
                <span className="metric-icon">📊</span>
              </div>
              <div className={`metric-value ${Math.abs(riskMetrics.beta - 1) < 0.2 ? 'positive' : 'warning'}`}>
                {riskMetrics.beta.toFixed(2)}
              </div>
              <div className="metric-description">与市场相关性</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">阿尔法系数</h3>
                <span className="metric-icon">🏆</span>
              </div>
              <div className={`metric-value ${riskMetrics.alpha >= 0 ? 'positive' : 'negative'}`}>
                {riskMetrics.alpha.toFixed(3)}
              </div>
              <div className="metric-description">超额收益能力</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-header">
                <h3 className="metric-title">波动率</h3>
                <span className="metric-icon">📊</span>
              </div>
              <div className={`metric-value ${riskMetrics.volatility < 0.2 ? 'positive' : 'warning'}`}>
                {(riskMetrics.volatility * 100).toFixed(2)}%
              </div>
              <div className="metric-description">收益波动程度</div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="dashboard-content">
        <div className="chart-section">
          <div className="chart-card">
            <div className="chart-header">
              <h2 className="chart-title">实验状态分布</h2>
              <Link to="/experiments" className="view-all-link">
                查看所有实验
              </Link>
            </div>
            <div className="chart-wrapper">
              <ReactECharts 
                option={pieChartOptions} 
                style={{ height: '400px', width: '100%' }}
                className="status-chart"
              />
            </div>
          </div>
        </div>
        
        <div className="recent-experiments-section">
          <div className="section-header">
            <h2>最近实验</h2>
            <Link to="/experiments" className="view-all-link">
              查看全部
            </Link>
          </div>
          
          {safeExperiments.length > 0 ? (
            <div className="recent-experiments-list">
              {safeExperiments
                .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
                .slice(0, 5)
                .map(experiment => (
                  <Link 
                    key={experiment.id} 
                    to={`/experiments/${experiment.id}`} 
                    className="recent-experiment-item"
                  >
                    <div className="experiment-info">
                      <h3 className="experiment-name">{experiment.name}</h3>
                      <div className="experiment-meta">
                        <div className={`experiment-status status-${experiment.status}`}>
                          {experiment.status}
                        </div>
                        <p className="created-at">
                          Created: {new Date(experiment.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="experiment-actions">
                      <span className="action-icon">→</span>
                    </div>
                  </Link>
                ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">📝</div>
              <h3>暂无实验</h3>
              <p>开始创建你的第一个实验</p>
              <Link to="/experiments" className="btn btn-primary">
                创建实验
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
