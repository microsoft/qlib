import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getExperiments, createExperiment, runExperiment, deleteExperiment, getExperimentLogs } from '../services/experiments'
import { getConfigs } from '../services/configs'
import YAMLEditor from '../components/YAMLEditor/YAMLEditor'
import ReactECharts from 'echarts-for-react'
import * as yaml from 'js-yaml'

interface Backtest {
  id: number
  name: string
  description: string
  config: any
  status: string
  created_at: string
  updated_at: string
  start_time?: string
  end_time?: string
  progress?: number
  performance?: any
  error?: string
}

interface Config {
  id: number
  name: string
  description: string
  content: string
  type: string
  created_at: string
  updated_at: string
}

const Backtest: React.FC = () => {
  const [backtests, setBacktests] = useState<Backtest[]>([])
  const [configs, setConfigs] = useState<Config[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [selectedConfig, setSelectedConfig] = useState<number | null>(null)
  const [yamlContent, setYamlContent] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [nameError, setNameError] = useState('')
  const [yamlError, setYamlError] = useState('')
  const [showLogs, setShowLogs] = useState<number | null>(null)
  const [logs, setLogs] = useState<{ [key: number]: string[] }>({})
  const [viewMode, setViewMode] = useState<'card' | 'list'>('list')
  const navigate = useNavigate()

  // 默认回测配置示例
  const defaultYamlExample = `# 回测配置示例
backtest:
  start_time: 2017-01-01
  end_time: 2020-08-01
  freq: day
  account: 100000000
  benchmark: SH000300
  exchange_kwargs:
    freq: day
    limit_threshold: 0.095
    deal_price: close
    open_cost: 0.0005
    close_cost: 0.0015
    min_cost: 5
strategy:
  class: TopkDropoutStrategy
  module_path: qlib.contrib.strategy.signal_strategy
  kwargs:
    topk: 50
    n_drop: 5
data:
  instruments: csi300
  start_time: 2008-01-01
  end_time: 2020-08-01
  freq: day
  handler:
    class: Alpha158
    module_path: qlib.contrib.data.handler
    kwargs:
      start_time: 2008-01-01
      end_time: 2020-08-01
      fit_start_time: 2008-01-01
      fit_end_time: 2014-12-31
      instruments: csi300
      freq: day
model:
  class: LGBModel
  module_path: qlib.contrib.model.gbdt
  kwargs:
    loss: mse
    colsample_bytree: 0.8879
    learning_rate: 0.0421
    subsample: 0.8789
    lambda_l1: 205.6999
    lambda_l2: 580.9768
    max_depth: 8
    num_leaves: 210
    num_threads: 20
`

  // Fetch backtests data with optimization to avoid unnecessary re-renders
  const fetchData = async () => {
    try {
      const experimentsData = await getExperiments()
      // Filter experiments to only show backtest-related ones
      const backtestExperiments = experimentsData.filter(exp => 
        exp.config?.backtest || exp.name?.toLowerCase().includes('backtest')
      )
      
      // Ensure backtestExperiments is an array and sort by created_at in descending order
      const sortedBacktests = Array.isArray(backtestExperiments) 
          ? backtestExperiments.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
          : []
      
      setBacktests(sortedBacktests)
    } catch (err) {
      console.error('Failed to fetch backtests:', err)
    }
  }

  // Fetch configs for backtest templates
  const fetchConfigs = async () => {
    try {
      const configsData = await getConfigs()
      // Filter configs to only show backtest templates
      const backtestConfigs = Array.isArray(configsData) ? 
        configsData.filter(config => ['backtest_template', 'experiment_template'].includes(config.type as string)) : []
      
      setConfigs(backtestConfigs)
    } catch (err) {
      console.error('Failed to fetch configs:', err)
      setConfigs([])
    } finally {
      setLoading(false)
    }
  }

  // Fetch data initially and then every 5 seconds for real-time updates
  useEffect(() => {
    fetchData()
    fetchConfigs()

    // Set up interval to refresh backtests every 5 seconds
    const interval = setInterval(fetchData, 5000)

    // Clean up interval on component unmount
    return () => clearInterval(interval)
  }, [])

  // Fetch logs for expanded backtests
  useEffect(() => {
    if (showLogs === null) return

    const fetchLogs = async () => {
      try {
        const logsContent = await getExperimentLogs(showLogs)
        if (logsContent) {
          // Split logs by newlines and filter out empty lines
          const logLines = logsContent.split('\n').filter(line => line.trim() !== '')
          setLogs(prev => ({
            ...prev,
            [showLogs]: logLines
          }))
        }
      } catch (err) {
        console.error('Failed to fetch logs:', err)
      }
    }

    // Fetch logs immediately and then every 3 seconds to reduce server load
    fetchLogs()
    const logsInterval = setInterval(fetchLogs, 3000)

    return () => clearInterval(logsInterval)
  }, [showLogs])

  const handleConfigChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const configId = e.target.value
    if (configId === '') {
      setSelectedConfig(null)
      setYamlContent(defaultYamlExample)
    } else {
      const id = parseInt(configId)
      setSelectedConfig(id)
      
      const config = configs.find(c => c.id === id)
      if (config) {
        setYamlContent(config.content)
        // 如果是回测模板，自动填充名称和描述
        if (config.type === 'backtest_template') {
          setName(config.name.replace(' Template', ''))
          setDescription(config.description || '')
        }
      }
    }
  }
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setYamlError('')
    
    // 客户端验证
    if (!name.trim()) {
      setNameError('回测名称不能为空');
      return;
    }
    
    if (name.length < 3) {
      setNameError('回测名称至少需要3个字符');
      return;
    }
    
    if (!yamlContent.trim()) {
      setYamlError('YAML配置不能为空');
      return;
    }
    
    setSubmitting(true)

    try {
      // Parse YAML content to JSON
      const config = yaml.load(yamlContent) as any
      
      // 验证YAML结构
      if (!config || typeof config !== 'object') {
        throw new Error('YAML配置必须是有效的对象');
      }
      
      if (!config.backtest) {
        throw new Error('YAML配置必须包含backtest字段');
      }
      
      await createExperiment({
        name,
        description,
        config
      })
      
      // Refresh backtests list
      fetchData()
      
      // Reset form
      resetForm()
    } catch (err: any) {
        if (err.name === 'YAMLException') {
          setYamlError(`YAML格式错误: ${err.message.split(' at line')[0]}`);
        } else if (err.response?.status === 403) {
          setError('您没有创建回测的权限，请联系管理员')
        } else {
          setError(err.response?.data?.detail || err.message || '创建回测失败');
        }
      } finally {
        setSubmitting(false)
      }
  }

  const handleRunBacktest = async (id: number) => {
    try {
      await runExperiment(id)
      
      // Immediately refresh backtests list to show pending status
      fetchData()
      
      // Set a short interval to refresh the list again after a delay to catch status changes
      const refreshInterval = setInterval(async () => {
        const latestData = await getExperiments()
        const backtestExperiments = latestData.filter(exp => 
          exp.config?.backtest || exp.name?.toLowerCase().includes('backtest')
        )
        setBacktests(Array.isArray(backtestExperiments) ? backtestExperiments : [])
      }, 2000)
      
      // Clear the interval after 10 seconds
      setTimeout(() => clearInterval(refreshInterval), 10000)
    } catch (err: any) {
      console.error('Failed to run backtest:', err)
      if (err.response?.status === 403) {
        alert('您没有运行回测的权限，请联系管理员')
      } else {
        alert('运行回测失败，请稍后重试')
      }
    }
  }

  const handleDeleteBacktest = async (id: number) => {
    if (window.confirm('确定要删除这个回测吗？')) {
      try {
        await deleteExperiment(id)
        // Refresh backtests list immediately after deletion
        fetchData()
      } catch (err: any) {
        console.error('Failed to delete backtest:', err)
        if (err.response?.status === 403) {
          alert('您没有删除回测的权限，请联系管理员')
        } else {
          alert('删除回测失败，请稍后重试')
        }
      }
    }
  }

  const resetForm = () => {
    setName('')
    setDescription('')
    setSelectedConfig(null)
    setYamlContent(defaultYamlExample)
    setShowForm(false)
    setError('')
    setNameError('')
    setYamlError('')
    setSubmitting(false)
  }

  // Prepare chart data if performance is available
  const getChartOption = (performance: any, chartType: 'cumulative' | 'drawdown' | 'monthly' = 'cumulative') => {
    if (!performance) {
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

    let title, data, dates, values, seriesType, yAxisFormatter, tooltipFormatter, seriesColor;

    switch (chartType) {
      case 'drawdown':
        title = '回撤曲线';
        data = performance.drawdown_curve || {};
        seriesType = 'line';
        seriesColor = '#ef4444';
        yAxisFormatter = '{value}%';
        tooltipFormatter = (params: any) => {
          const date = params[0].axisValue;
          const value = params[0].value;
          return `${date}<br/>Drawdown: ${(value * 100).toFixed(2)}%`;
        };
        break;
      case 'monthly':
        title = '月度收益';
        data = performance.monthly_returns || {};
        seriesType = 'bar';
        seriesColor = '#10b981';
        yAxisFormatter = '{value}%';
        tooltipFormatter = (params: any) => {
          const date = params[0].axisValue;
          const value = params[0].value;
          return `${date}<br/>Monthly Return: ${(value * 100).toFixed(2)}%`;
        };
        break;
      case 'cumulative':
      default:
        title = '累计收益曲线';
        data = performance.cumulative_returns || {};
        seriesType = 'line';
        seriesColor = '#646cff';
        yAxisFormatter = '{value}%';
        tooltipFormatter = (params: any) => {
          const date = params[0].axisValue;
          const value = params[0].value;
          return `${date}<br/>Cumulative Return: ${(value * 100).toFixed(2)}%`;
        };
        break;
    }

    dates = Object.keys(data);
    values = Object.values(data) as number[];

    return {
      title: {
        text: title,
        left: 'center',
        textStyle: {
          fontSize: 14
        }
      },
      tooltip: {
        trigger: 'axis',
        formatter: tooltipFormatter
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          rotate: chartType === 'monthly' ? 0 : 45,
          fontSize: 10
        }
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: yAxisFormatter,
          fontSize: 10
        }
      },
      series: [
        {
          data: values.map(v => (v * 100).toFixed(2)),
          type: seriesType,
          smooth: seriesType === 'line',
          itemStyle: {
            color: seriesColor,
            ...(seriesType === 'bar' && {
              color: (params: any) => {
                const value = parseFloat(params.value);
                return value >= 0 ? '#10b981' : '#ef4444';
              }
            })
          },
          ...(seriesType === 'line' && {
            areaStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: seriesColor + '80' },
                  { offset: 1, color: seriesColor + '10' }
                ]
              }
            }
          })
        }
      ]
    }
  }

  if (loading) {
    return <div className="container">Loading...</div>
  }

  return (
    <div className="container page-transition">
      <div className="page-header">
        <h1>回测管理</h1>
        <button className="btn" onClick={() => setShowForm(true)}>
          创建回测
        </button>
      </div>

      {showForm && (
        <div className="card" style={{ marginBottom: '20px' }}>
          <h2>创建回测</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="name">名称</label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => {
                  const value = e.target.value;
                  setName(value);
                  // 实时验证名称
                  if (!value.trim()) {
                    setNameError('回测名称不能为空');
                  } else if (value.length < 3) {
                    setNameError('回测名称至少需要3个字符');
                  } else {
                    setNameError('');
                  }
                }}
                placeholder="例如: Alpha158_LGB_2017-2020"
                required
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border: `1px solid ${nameError ? '#ff4d4f' : '#d9d9d9'}`,
                  fontSize: '14px',
                  transition: 'all 0.3s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => {
                  (e.target as HTMLInputElement).style.borderColor = nameError ? '#ff4d4f' : '#1890ff';
                  (e.target as HTMLInputElement).style.boxShadow = nameError ? '0 0 0 2px rgba(255, 77, 79, 0.2)' : '0 0 0 2px rgba(24, 144, 255, 0.2)';
                }}
                onBlur={(e) => {
                  (e.target as HTMLInputElement).style.borderColor = nameError ? '#ff4d4f' : '#d9d9d9';
                  (e.target as HTMLInputElement).style.boxShadow = 'none';
                }}
              />
              {nameError && (
                <small style={{ color: '#ff4d4f', display: 'block', marginTop: '6px', fontSize: '12px' }}>
                  ⚠️ {nameError}
                </small>
              )}
              {!nameError && (
                <small style={{ color: '#666', display: 'block', marginTop: '6px', fontSize: '12px' }}>
                  回测的唯一标识符，用于区分不同回测
                </small>
              )}
            </div>
            <div className="form-group">
              <label htmlFor="description">描述</label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                placeholder="例如: 使用LightGBM模型和Alpha158因子进行回测实验"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border: '1px solid #d9d9d9',
                  fontSize: '14px',
                  resize: 'vertical',
                  transition: 'all 0.3s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => {
                  (e.target as HTMLTextAreaElement).style.borderColor = '#1890ff';
                  (e.target as HTMLTextAreaElement).style.boxShadow = '0 0 0 2px rgba(24, 144, 255, 0.2)';
                }}
                onBlur={(e) => {
                  (e.target as HTMLTextAreaElement).style.borderColor = '#d9d9d9';
                  (e.target as HTMLTextAreaElement).style.boxShadow = 'none';
                }}
              />
              <small style={{ color: '#666', display: 'block', marginTop: '6px', fontSize: '12px' }}>
                回测的详细描述，帮助您和团队理解回测目的
              </small>
            </div>
            
            <div className="form-group">
              <label htmlFor="config">选择回测模板</label>
              <select
                id="config"
                value={selectedConfig || ''}
                onChange={handleConfigChange}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border: '1px solid #d9d9d9',
                  fontSize: '14px',
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  boxSizing: 'border-box',
                  appearance: 'none',
                  backgroundImage: 'url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1024 1024\" width=\"16\" height=\"16\"%3E%3Cpath fill=\"%23666\" d=\"M840.4 300H183.6c-19.7 0-35.3 15.6-35.3 35.3v35.3c0 19.7 15.6 35.3 35.3 35.3h656.8c19.7 0 35.3-15.6 35.3-35.3v-35.3c0-19.7-15.6-35.3-35.3-35.3z\"/%3E%3C/svg%3E")',
                  backgroundRepeat: 'no-repeat',
                  backgroundPosition: 'right 12px center',
                  paddingRight: '32px'
                }}
                onFocus={(e) => {
                  (e.target as HTMLSelectElement).style.borderColor = '#1890ff';
                  (e.target as HTMLSelectElement).style.boxShadow = '0 0 0 2px rgba(24, 144, 255, 0.2)';
                }}
                onBlur={(e) => {
                  (e.target as HTMLSelectElement).style.borderColor = '#d9d9d9';
                  (e.target as HTMLSelectElement).style.boxShadow = 'none';
                }}
              >
                <option value="">-- 选择模板或使用默认配置 --</option>
                {configs.map(config => (
                  <option key={config.id} value={config.id}>
                    {config.name}
                  </option>
                ))}
              </select>
              <small style={{ color: '#666', display: 'block', marginTop: '6px', fontSize: '12px' }}>
                选择一个预定义的回测模板，或直接在下方编辑器中输入自定义配置
              </small>
            </div>
            <div className="form-group">
              <label htmlFor="yamlContent">回测配置</label>
              <div style={{ marginBottom: '12px', padding: '12px', backgroundColor: '#f9f9f9', borderRadius: '6px', border: '1px solid #e8e8e8' }}>
                <strong style={{ fontSize: '13px', color: '#333' }}>配置说明：</strong>
                <ul style={{ margin: '6px 0 0 20px', fontSize: '12px', color: '#666', lineHeight: '1.5' }}>
                  <li>使用YAML格式定义回测配置</li>
                  <li>必须包含 <code>backtest</code> 字段，定义回测基本参数</li>
                  <li>必须包含 <code>strategy</code> 字段，定义交易策略</li>
                  <li>必须包含 <code>data</code> 字段，定义数据来源</li>
                  <li>必须包含 <code>model</code> 字段，定义预测模型</li>
                  <li>可使用预定义模板或修改默认示例</li>
                </ul>
              </div>
              <YAMLEditor
                value={yamlContent || defaultYamlExample}
                onChange={setYamlContent}
                error={yamlError}
              />
              {yamlError && (
                <small style={{ color: '#ff4d4f', display: 'block', marginTop: '6px', fontSize: '12px' }}>
                  ⚠️ {yamlError}
                </small>
              )}
            </div>
            {error && (
              <div style={{ marginBottom: '15px', padding: '10px 12px', backgroundColor: '#fff1f0', color: '#ff4d4f', borderRadius: '6px', border: '1px solid #ffccc7', fontSize: '13px' }}>
                ⚠️ {error}
              </div>
            )}
            <div className="form-actions">
              <button 
                type="submit" 
                className="btn"
                disabled={submitting}
                style={{
                  padding: '10px 24px',
                  backgroundColor: submitting ? '#40a9ff' : '#1890ff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: submitting ? 'not-allowed' : 'pointer',
                  transition: 'all 0.3s',
                  opacity: submitting ? 0.8 : 1
                }}
              >
                {submitting ? '创建中...' : '创建回测'}
              </button>
              <button type="button" className="btn" onClick={resetForm}>
                取消
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="experiments-list-header">
        <h2>回测列表</h2>
        <div className="view-mode-switcher">
          <button 
            className={`view-btn ${viewMode === 'card' ? 'active' : ''}`}
            onClick={() => setViewMode('card')}
          >
            卡片视图
          </button>
          <button 
            className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
            onClick={() => setViewMode('list')}
          >
            列表视图
          </button>
        </div>
      </div>

      <div className={`experiments-list view-${viewMode}`}>
        {backtests.map(backtest => (
          <div key={backtest.id} className={viewMode === 'card' ? 'experiment-card' : 'experiment-list-item'}>
            {viewMode === 'card' ? (
              <>
                <div className="experiment-header">
                  <h3 className="experiment-name">{backtest.name}</h3>
                  <span className={`experiment-status status-${backtest.status}`}>
                    {backtest.status === 'created' && '已创建'}
                    {backtest.status === 'pending' && '待运行'}
                    {backtest.status === 'running' && '运行中'}
                    {backtest.status === 'completed' && '已完成'}
                    {backtest.status === 'failed' && '失败'}
                    {backtest.status === 'stopped' && '已停止'}
                    {!['created', 'pending', 'running', 'completed', 'failed', 'stopped'].includes(backtest.status) && backtest.status}
                  </span>
                </div>
                
                <p className="experiment-description">{backtest.description}</p>
                
                <div className="experiment-meta">
                  <div className="meta-item">
                    <span className="meta-label">创建时间:</span>
                    <span className="meta-value">{new Date(backtest.created_at).toLocaleString()}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">更新时间:</span>
                    <span className="meta-value">{new Date(backtest.updated_at).toLocaleString()}</span>
                  </div>
                  {backtest.start_time && (
                    <div className="meta-item">
                      <span className="meta-label">开始时间:</span>
                      <span className="meta-value">{new Date(backtest.start_time).toLocaleString()}</span>
                    </div>
                  )}
                  {backtest.end_time && (
                    <div className="meta-item">
                      <span className="meta-label">结束时间:</span>
                      <span className="meta-value">{new Date(backtest.end_time).toLocaleString()}</span>
                    </div>
                  )}
                  {backtest.progress !== undefined && (
                    <div className="meta-item full-width">
                      <span className="meta-label">进度:</span>
                      <div className="progress-bar-container">
                        <div 
                          className="progress-bar" 
                          style={{ width: `${backtest.progress}%` }}
                        ></div>
                        <span className="progress-text">{backtest.progress.toFixed(0)}%</span>
                      </div>
                    </div>
                  )}
                </div>
                
                {backtest.performance && (
                  <div className="experiment-performance">
                    <h4>回测结果</h4>
                    <div className="performance-metrics">
                      <div className="metrics-grid">
                        {/* 核心指标 */}
                        <div className="metrics-section">
                          <h5>核心指标</h5>
                          <div className="metrics-row">
                            {backtest.performance.total_return !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">总收益</span>
                                <span className={`metric-value ${backtest.performance.total_return >= 0 ? 'positive' : 'negative'}`}>
                                  {(backtest.performance.total_return * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {backtest.performance.annual_return !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">年化收益</span>
                                <span className={`metric-value ${backtest.performance.annual_return >= 0 ? 'positive' : 'negative'}`}>
                                  {(backtest.performance.annual_return * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {backtest.performance.sharpe_ratio !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">夏普比率</span>
                                <span className={`metric-value ${backtest.performance.sharpe_ratio >= 0 ? 'positive' : 'negative'}`}>
                                  {backtest.performance.sharpe_ratio.toFixed(2)}
                                </span>
                              </div>
                            )}
                            {backtest.performance.max_drawdown !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">最大回撤</span>
                                <span className="metric-value">
                                  {(backtest.performance.max_drawdown * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        {/* 交易指标 */}
                        <div className="metrics-section">
                          <h5>交易指标</h5>
                          <div className="metrics-row">
                            {backtest.performance.win_rate !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">胜率</span>
                                <span className={`metric-value ${backtest.performance.win_rate >= 0.5 ? 'positive' : 'negative'}`}>
                                  {(backtest.performance.win_rate * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {backtest.performance.avg_win !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">平均盈利</span>
                                <span className={`metric-value ${backtest.performance.avg_win >= 0 ? 'positive' : 'negative'}`}>
                                  {(backtest.performance.avg_win * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {backtest.performance.avg_loss !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">平均亏损</span>
                                <span className={`metric-value ${backtest.performance.avg_loss >= 0 ? 'positive' : 'negative'}`}>
                                  {(backtest.performance.avg_loss * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      {/* 收益分布 */}
                      {backtest.performance.return_distribution && (
                        <div className="metrics-section">
                          <h5>收益分布</h5>
                          <div className="return-distribution">
                            <div className="distribution-item">
                              <span className="dist-key">最小值</span>
                              <span className="dist-value">{(backtest.performance.return_distribution.min * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">25%分位数</span>
                              <span className="dist-value">{(backtest.performance.return_distribution['25th'] * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">中位数</span>
                              <span className="dist-value">{(backtest.performance.return_distribution.median * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">75%分位数</span>
                              <span className="dist-value">{(backtest.performance.return_distribution['75th'] * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">最大值</span>
                              <span className="dist-value">{(backtest.performance.return_distribution.max * 100).toFixed(2)}%</span>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Performance Charts */}
                      <div className="performance-charts">
                        <h5>回测结果图表</h5>
                        <div className="charts-container">
                          {/* Cumulative Returns Chart */}
                          <div className="chart-item">
                            <h6>累计收益曲线</h6>
                            <div className="chart-wrapper" style={{ height: '250px' }}>
                              <ReactECharts option={getChartOption(backtest.performance, 'cumulative')} style={{ height: '100%' }} />
                            </div>
                          </div>
                          
                          {/* Drawdown Curve Chart */}
                          {backtest.performance.drawdown_curve && Object.keys(backtest.performance.drawdown_curve).length > 0 && (
                            <div className="chart-item">
                              <h6>回撤曲线</h6>
                              <div className="chart-wrapper" style={{ height: '250px' }}>
                                <ReactECharts option={getChartOption(backtest.performance, 'drawdown')} style={{ height: '100%' }} />
                              </div>
                            </div>
                          )}
                          
                          {/* Monthly Returns Chart */}
                          {backtest.performance.monthly_returns && Object.keys(backtest.performance.monthly_returns).length > 0 && (
                            <div className="chart-item">
                              <h6>月度收益</h6>
                              <div className="chart-wrapper" style={{ height: '250px' }}>
                                <ReactECharts option={getChartOption(backtest.performance, 'monthly')} style={{ height: '100%' }} />
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {backtest.error && (
                  <div className="experiment-error">
                    <h4>Error</h4>
                    <p className="error-message">{backtest.error}</p>
                  </div>
                )}
                
                <div className="experiment-actions">
                  <button 
                      className="action-btn view-btn"
                      onClick={() => navigate(`/experiments/${backtest.id}`)}
                    >
                      查看详情
                    </button>
                    {(backtest.status === 'created' || backtest.status === 'completed' || backtest.status === 'failed') && (
                    <button 
                      className="action-btn run-btn"
                      onClick={() => handleRunBacktest(backtest.id)}
                    >
                      {backtest.status === 'created' ? '运行回测' : '重新运行'}
                    </button>
                  )}
                  <button 
                    className="action-btn logs-btn"
                    onClick={() => setShowLogs(showLogs === backtest.id ? null : backtest.id)}
                  >
                    {showLogs === backtest.id ? '隐藏日志' : '查看日志'}
                  </button>
                  <button 
                    className="action-btn delete-btn"
                    onClick={() => handleDeleteBacktest(backtest.id)}
                  >
                    删除
                  </button>
                </div>
                
                {showLogs === backtest.id && (
                  <div className="experiment-logs">
                    <h4>运行日志</h4>
                    <div className="logs-container">
                      {logs[backtest.id] && logs[backtest.id].length > 0 ? (
                        <div className="logs-content">
                          {logs[backtest.id].map((log, index) => (
                            <div key={index} className="log-line">
                              {log}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="no-logs">
                          {backtest.status === 'running' ? '正在运行...' : '暂无日志'}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="experiment-list-content">
                <div className="experiment-list-header">
                  <h3 className="experiment-name">{backtest.name}</h3>
                  <div className="header-right">
                    <span className={`experiment-status status-${backtest.status}`}>
                      {backtest.status === 'created' && '已创建'}
                      {backtest.status === 'pending' && '待运行'}
                      {backtest.status === 'running' && '运行中'}
                      {backtest.status === 'completed' && '已完成'}
                      {backtest.status === 'failed' && '失败'}
                      {backtest.status === 'stopped' && '已停止'}
                      {!['created', 'pending', 'running', 'completed', 'failed', 'stopped'].includes(backtest.status) && backtest.status}
                    </span>
                    <div className="meta-item created-time">
                      <span className="meta-label">创建时间:</span>
                      <span className="meta-value">{new Date(backtest.created_at).toLocaleString()}</span>
                    </div>
                  </div>
                </div>
                <p className="experiment-description">{backtest.description}</p>
                {backtest.progress !== undefined && (
                  <div className="experiment-meta">
                    <div className="meta-item">
                      <span className="meta-label">进度:</span>
                      <span className="progress-text">{backtest.progress.toFixed(0)}%</span>
                    </div>
                  </div>
                )}
                <div className="experiment-actions">
                  <button 
                    className="action-btn view-btn"
                    onClick={() => navigate(`/experiments/${backtest.id}`)}
                  >
                    查看详情
                  </button>
                  {(backtest.status === 'created' || backtest.status === 'completed' || backtest.status === 'failed') && (
                    <button 
                      className="action-btn run-btn"
                      onClick={() => handleRunBacktest(backtest.id)}
                    >
                      {backtest.status === 'created' ? '运行回测' : '重新运行'}
                    </button>
                  )}
                  <button 
                    className="action-btn logs-btn"
                    onClick={() => setShowLogs(showLogs === backtest.id ? null : backtest.id)}
                  >
                    {showLogs === backtest.id ? '隐藏日志' : '查看日志'}
                  </button>
                  <button 
                    className="action-btn delete-btn"
                    onClick={() => handleDeleteBacktest(backtest.id)}
                  >
                    删除
                  </button>
                </div>
                {showLogs === backtest.id && (
                  <div className="experiment-logs">
                    <h4>运行日志</h4>
                    <div className="logs-container">
                      {logs[backtest.id] && logs[backtest.id].length > 0 ? (
                        <div className="logs-content">
                          {logs[backtest.id].map((log, index) => (
                            <div key={index} className="log-line">
                              {log}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="no-logs">
                          {backtest.status === 'running' ? '正在运行...' : '暂无日志'}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default Backtest
