import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getExperiments, createExperiment, runExperiment, deleteExperiment, getExperimentLogs } from '../services/experiments'
import { getConfigs } from '../services/configs'
import { getBenchmarks } from '../services/benchmarks'
import { getUserInfo } from '../services/auth'
import type { ConfigType } from '../services/configs'
import YAMLEditor from '../components/YAMLEditor/YAMLEditor'
import ReactECharts from 'echarts-for-react'
import * as yaml from 'js-yaml'

interface Experiment {
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
  type: ConfigType
  created_at: string
  updated_at: string
}

const Experiments: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [configs, setConfigs] = useState<Config[]>([])
  const [benchmarks, setBenchmarks] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [selectedConfig, setSelectedConfig] = useState<number | null>(null)
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null)
  const [yamlContent, setYamlContent] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [nameError, setNameError] = useState('')
  const [yamlError, setYamlError] = useState('')
  const [userInfo, setUserInfo] = useState<any>(null)
  const [showLogs, setShowLogs] = useState<number | null>(null)
  const [logs, setLogs] = useState<{ [key: number]: string[] }>({})
  const [viewMode, setViewMode] = useState<'card' | 'list'>('list')
  const [selectedExperiments, setSelectedExperiments] = useState<number[]>([])
  const [showComparison, setShowComparison] = useState(false)
  const navigate = useNavigate()

  // 默认YAML配置示例
  const defaultYamlExample = `# QLib实验配置示例
qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn
market: &market csi300
benchmark: &benchmark SH000300
data_handler_config:
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market
    freq: day
task:
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
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs:
                    start_time: 2008-01-01
                    end_time: 2020-08-01
                    fit_start_time: 2008-01-01
                    fit_end_time: 2014-12-31
                    instruments: *market
                    freq: day
            segments:
                train:
                    - "2008-01-01"
                    - "2014-12-31"
                valid:
                    - "2015-01-01"
                    - "2016-12-31"
                test:
                    - "2017-01-01"
                    - "2020-08-01"
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              config:
                  strategy:
                      class: TopkDropoutStrategy
                      module_path: qlib.contrib.strategy.signal_strategy
                      kwargs:
                          topk: 50
                          n_drop: 5
                  backtest:
                      start_time: 2017-01-01
                      end_time: 2020-08-01
                      freq: day
                      account: 100000000
                      benchmark: *benchmark
                      exchange_kwargs:
                          freq: day
                          limit_threshold: 0.095
                          deal_price: close
                          open_cost: 0.0005
                          close_cost: 0.0015
                          min_cost: 5
`

  // Get user info from localStorage and API
  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        // First try to get from API
        const info = await getUserInfo()
        if (info) {
          setUserInfo(info)
          localStorage.setItem('userInfo', JSON.stringify(info))
        } else {
          // Fallback to localStorage
          const storedUserInfo = localStorage.getItem('userInfo')
          if (storedUserInfo) {
            setUserInfo(JSON.parse(storedUserInfo))
          }
        }
      } catch (error) {
        console.error('Failed to get user info:', error)
        // Fallback to localStorage
        const storedUserInfo = localStorage.getItem('userInfo')
        if (storedUserInfo) {
          setUserInfo(JSON.parse(storedUserInfo))
        }
      }
    }
    
    fetchUserInfo()
  }, [])
  
  // For now, always show the create experiment button
  // The actual permission check will be handled by the backend
  const canCreateExperiments = true
  
  // Debug: Log user info and permission status
  useEffect(() => {
    console.log('User Info:', userInfo)
    console.log('Can Create Experiments:', canCreateExperiments)
  }, [userInfo, canCreateExperiments])

  // Fetch experiments data with optimization to avoid unnecessary re-renders
  const fetchData = async () => {
    try {
      const experimentsData = await getExperiments()
      // Ensure experimentsData is an array and sort by created_at in descending order
      const sortedExperiments = Array.isArray(experimentsData) 
          ? experimentsData.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
          : []
      
      // Only update state if data has changed to avoid unnecessary re-renders
      if (JSON.stringify(sortedExperiments) !== JSON.stringify(experiments)) {
        setExperiments(sortedExperiments)
      }
    } catch (err) {
      console.error('Failed to fetch experiments:', err)
    }
  }

  // Fetch data initially and then every 5 seconds for real-time updates
  useEffect(() => {
      // Fetch configs only once initially
      const fetchConfigs = async () => {
          try {
              const configsData = await getConfigs()
              // Ensure configsData is an array
              setConfigs(Array.isArray(configsData) ? configsData : [])
          } catch (err) {
              console.error('Failed to fetch configs:', err)
              // Set empty array on error
              setConfigs([])
          }
      }

      // Fetch benchmarks only once initially
      const fetchBenchmarks = async () => {
          try {
              const benchmarksData = await getBenchmarks()
              setBenchmarks(benchmarksData)
          } catch (err) {
              console.error('Failed to fetch benchmarks:', err)
              // Set empty array on error
              setBenchmarks([])
          } finally {
              setLoading(false)
          }
      }

      fetchData()
      fetchConfigs()
      fetchBenchmarks()

      // Set up interval to refresh experiments every 5 seconds
      const interval = setInterval(fetchData, 5000)

      // Clean up interval on component unmount
      return () => clearInterval(interval)
  }, [fetchData])

  // Fetch logs for expanded experiments every 1 second for better real-time updates
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

  // Auto-expand logs when experiment status changes to running
  useEffect(() => {
    // Only auto-expand logs if user hasn't manually selected any logs
    if (showLogs !== null) return
    
    if (experiments.length > 0) {
      const runningExperiments = experiments.filter(exp => exp.status === 'running' || exp.status === 'pending')
      if (runningExperiments.length > 0) {
        // Only auto-expand logs for the first running experiment
        const firstRunningExp = runningExperiments[0]
        setShowLogs(firstRunningExp.id)
      }
    }
  }, [experiments, showLogs])

  const handleConfigChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const configId = e.target.value
    if (configId === '') {
      setSelectedConfig(null)
      setYamlContent(defaultYamlExample)
    } else {
      const id = parseInt(configId)
      setSelectedConfig(id)
      setSelectedBenchmark(null)
      
      const config = configs.find(c => c.id === id)
      if (config) {
        setYamlContent(config.content)
        // 如果是实验模板，自动填充名称和描述
        if (config.type === 'experiment_template') {
          setName(config.name.replace(' Template', ''))
          setDescription(config.description || '')
        }
      }
    }
  }
  
  const handleBenchmarkChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const benchmarkId = e.target.value
    setSelectedBenchmark(benchmarkId)
    setSelectedConfig(null)
    
    const benchmark = benchmarks.find(b => b.id === benchmarkId)
    if (benchmark) {
      setYamlContent(benchmark.content)
      // 自动填充名称和描述
      setName(benchmark.name)
      setDescription(`基于${benchmark.model}模型的benchmark实验`)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setYamlError('')
    
    // 客户端验证
    if (!name.trim()) {
      setNameError('实验名称不能为空');
      return;
    }
    
    if (name.length < 3) {
      setNameError('实验名称至少需要3个字符');
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
      
      if (!config.task) {
        throw new Error('YAML配置必须包含task字段');
      }
      
      if (!config.task.model || !config.task.dataset) {
        throw new Error('task字段必须包含model和dataset');
      }
      
      await createExperiment({
        name,
        description,
        config
      })
      
      // Refresh experiments list
      const experimentsData = await getExperiments()
      setExperiments(experimentsData)
      
      // Reset form
      resetForm()
    } catch (err: any) {
        if (err.name === 'YAMLException') {
          setYamlError(`YAML格式错误: ${err.message.split(' at line')[0]}`);
        } else if (err.response?.status === 403) {
          setError('您没有创建实验的权限，请联系管理员')
        } else {
          setError(err.response?.data?.detail || err.message || '创建实验失败');
        }
      } finally {
        setSubmitting(false)
      }
  }

  const handleRunExperiment = async (id: number) => {
    try {
      await runExperiment(id)
      
      // Immediately refresh experiments list to show pending status
      const experimentsData = await getExperiments()
      setExperiments(experimentsData)
      
      // Set a short interval to refresh the list again after a delay to catch status changes
      const refreshInterval = setInterval(async () => {
        const latestData = await getExperiments()
        setExperiments(latestData)
      }, 2000)
      
      // Clear the interval after 10 seconds
      setTimeout(() => clearInterval(refreshInterval), 10000)
    } catch (err: any) {
      console.error('Failed to run experiment:', err)
      if (err.response?.status === 403) {
        alert('您没有运行实验的权限，请联系管理员')
      } else {
        alert('运行实验失败，请稍后重试')
      }
    }
  }

  const handleDeleteExperiment = async (id: number) => {
    if (window.confirm('确定要删除这个实验吗？')) {
      try {
        await deleteExperiment(id)
        // Refresh experiments list immediately after deletion
        await fetchData()
      } catch (err: any) {
        console.error('Failed to delete experiment:', err)
        if (err.response?.status === 403) {
          alert('您没有删除实验的权限，请联系管理员')
        } else {
          alert('删除实验失败，请稍后重试')
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

  // 处理实验选择
  const handleExperimentSelect = (id: number) => {
    setSelectedExperiments(prev => {
      if (prev.includes(id)) {
        return prev.filter(expId => expId !== id)
      } else {
        // 最多选择5个实验进行对比
        if (prev.length >= 5) {
          alert('最多只能选择5个实验进行对比')
          return prev
        }
        return [...prev, id]
      }
    })
  }

  // 清除选择的实验
  const clearSelectedExperiments = () => {
    setSelectedExperiments([])
    setShowComparison(false)
  }

  // 生成实验对比图表
  const getComparisonChartOption = () => {
    const selectedExps = experiments.filter(exp => selectedExperiments.includes(exp.id))
    
    if (selectedExps.length < 2) {
      return {
        title: {
          text: '请至少选择2个实验进行对比',
          left: 'center'
        }
      }
    }
    
    // 提取所有实验的累计收益曲线
    const allDates = new Set<string>()
    const series = selectedExps.map(exp => {
      if (exp.performance && exp.performance.cumulative_returns) {
        const dates = Object.keys(exp.performance.cumulative_returns)
        dates.forEach(date => allDates.add(date))
        return {
          name: exp.name,
          type: 'line',
          data: dates.map(date => (exp.performance.cumulative_returns[date] * 100).toFixed(2)),
          smooth: true,
          lineStyle: {
            width: 2
          },
          areaStyle: {
            opacity: 0.1
          }
        }
      }
      return {
        name: exp.name,
        type: 'line',
        data: [],
        smooth: true
      }
    })
    
    const sortedDates = Array.from(allDates).sort()
    
    return {
      title: {
        text: '实验收益对比',
        left: 'center',
        textStyle: {
          fontSize: 14
        }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          let result = `${params[0].axisValue}<br/>`
          params.forEach((param: any) => {
            result += `${param.seriesName}: ${param.value}%<br/>`
          })
          return result
        }
      },
      legend: {
        data: selectedExps.map(exp => exp.name),
        orient: 'horizontal',
        bottom: 10,
        textStyle: {
          fontSize: 12
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: sortedDates,
        axisLabel: {
          rotate: 45,
          fontSize: 10
        }
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}%',
          fontSize: 12
        }
      },
      series: series
    }
  }

  // 生成实验指标对比表格
  const renderComparisonTable = () => {
    const selectedExps = experiments.filter(exp => selectedExperiments.includes(exp.id))
    
    if (selectedExps.length < 2) {
      return <div className="empty-comparison">请至少选择2个实验进行对比</div>
    }
    
    return (
      <div className="comparison-table-container">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>指标</th>
              {selectedExps.map(exp => (
                <th key={exp.id}>{exp.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>总收益</td>
              {selectedExps.map(exp => (
                <td key={exp.id} className={exp.performance?.total_return >= 0 ? 'positive' : 'negative'}>
                  {exp.performance?.total_return !== undefined ? `${(exp.performance.total_return * 100).toFixed(2)}%` : '-'}
                </td>
              ))}
            </tr>
            <tr>
              <td>年化收益</td>
              {selectedExps.map(exp => (
                <td key={exp.id} className={exp.performance?.annual_return >= 0 ? 'positive' : 'negative'}>
                  {exp.performance?.annual_return !== undefined ? `${(exp.performance.annual_return * 100).toFixed(2)}%` : '-'}
                </td>
              ))}
            </tr>
            <tr>
              <td>夏普比率</td>
              {selectedExps.map(exp => (
                <td key={exp.id} className={exp.performance?.sharpe_ratio >= 1 ? 'positive' : exp.performance?.sharpe_ratio > 0 ? 'warning' : 'negative'}>
                  {exp.performance?.sharpe_ratio !== undefined ? exp.performance.sharpe_ratio.toFixed(2) : '-'}
                </td>
              ))}
            </tr>
            <tr>
              <td>最大回撤</td>
              {selectedExps.map(exp => (
                <td key={exp.id}>
                  {exp.performance?.max_drawdown !== undefined ? `${(exp.performance.max_drawdown * 100).toFixed(2)}%` : '-'}
                </td>
              ))}
            </tr>
            <tr>
              <td>胜率</td>
              {selectedExps.map(exp => (
                <td key={exp.id} className={exp.performance?.win_rate >= 0.5 ? 'positive' : 'negative'}>
                  {exp.performance?.win_rate !== undefined ? `${(exp.performance.win_rate * 100).toFixed(2)}%` : '-'}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    )
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
        <h1>实验管理</h1>
        {canCreateExperiments && (
          <button className="btn" onClick={() => setShowForm(true)}>
            创建实验
          </button>
        )}
      </div>

      {showForm && (
        <div className="card" style={{ marginBottom: '20px' }}>
          <h2>创建实验</h2>
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
                    setNameError('实验名称不能为空');
                  } else if (value.length < 3) {
                    setNameError('实验名称至少需要3个字符');
                  } else {
                    setNameError('');
                  }
                }}
                placeholder="例如: LightGBM_Alpha158_Experiment"
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
                  实验的唯一标识符，用于区分不同实验
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
                实验的详细描述，帮助您和团队理解实验目的
              </small>
            </div>
            <div className="form-group">
              <label htmlFor="benchmark">选择Benchmark样例</label>
              <select
                id="benchmark"
                value={selectedBenchmark || ''}
                onChange={handleBenchmarkChange}
              >
                <option value="">-- 选择Benchmark样例 --</option>
                {benchmarks.map(benchmark => (
                  <option key={benchmark.id} value={benchmark.id}>
                    {benchmark.name}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="config">或选择配置模板</label>
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
                选择一个预定义的配置模板，或直接在下方编辑器中输入自定义配置
              </small>
            </div>
            <div className="form-group">
              <label htmlFor="yamlContent">YAML配置</label>
              <div style={{ marginBottom: '12px', padding: '12px', backgroundColor: '#f9f9f9', borderRadius: '6px', border: '1px solid #e8e8e8' }}>
                <strong style={{ fontSize: '13px', color: '#333' }}>配置说明：</strong>
                <ul style={{ margin: '6px 0 0 20px', fontSize: '12px', color: '#666', lineHeight: '1.5' }}>
                  <li>使用YAML格式定义实验配置</li>
                  <li>必须包含 <code>task</code> 字段，其中包含 <code>model</code> 和 <code>dataset</code></li>
                  <li>model和dataset必须指定 <code>class</code> 和 <code>module_path</code></li>
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
                {submitting ? '创建中...' : '创建实验'}
              </button>
              <button type="button" className="btn" onClick={resetForm}>
                取消
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="experiments-list-header">
        <h2>实验列表</h2>
        <div className="header-controls">
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
          {selectedExperiments.length > 0 && (
            <div className="comparison-controls">
              <span className="selected-count">已选择 {selectedExperiments.length} 个实验</span>
              <button 
                className="btn btn-secondary" 
                onClick={clearSelectedExperiments}
              >
                清除选择
              </button>
              {selectedExperiments.length >= 2 && (
                <button 
                  className="btn btn-primary" 
                  onClick={() => setShowComparison(!showComparison)}
                >
                  {showComparison ? '隐藏对比' : '开始对比'}
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {showComparison && (
        <div className="experiment-comparison-section">
          <h2>实验对比</h2>
          <div className="comparison-chart">
            <ReactECharts option={getComparisonChartOption()} style={{ height: '400px' }} />
          </div>
          {renderComparisonTable()}
        </div>
      )}
      
      <div className={`experiments-list view-${viewMode}`}>
        {experiments.map(experiment => (
          <div key={experiment.id} className={viewMode === 'card' ? 'experiment-card' : 'experiment-list-item'}>
            <div className="experiment-selector">
              <input
                type="checkbox"
                checked={selectedExperiments.includes(experiment.id)}
                onChange={() => handleExperimentSelect(experiment.id)}
                className="experiment-checkbox"
              />
            </div>
            {viewMode === 'card' ? (
              <>
                <div className="experiment-header">
                  <h3 className="experiment-name">{experiment.name}</h3>
                  <span className={`experiment-status status-${experiment.status}`}>
                    {experiment.status === 'created' && '已创建'}
                    {experiment.status === 'pending' && '待运行'}
                    {experiment.status === 'running' && '运行中'}
                    {experiment.status === 'completed' && '已完成'}
                    {experiment.status === 'failed' && '失败'}
                    {experiment.status === 'stopped' && '已停止'}
                    {!['created', 'pending', 'running', 'completed', 'failed', 'stopped'].includes(experiment.status) && experiment.status}
                  </span>
                </div>
                
                <p className="experiment-description">{experiment.description}</p>
                
                <div className="experiment-meta">
                  <div className="meta-item">
                    <span className="meta-label">创建时间:</span>
                    <span className="meta-value">{new Date(experiment.created_at).toLocaleString()}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">更新时间:</span>
                    <span className="meta-value">{new Date(experiment.updated_at).toLocaleString()}</span>
                  </div>
                  {experiment.start_time && (
                    <div className="meta-item">
                      <span className="meta-label">开始时间:</span>
                      <span className="meta-value">{new Date(experiment.start_time).toLocaleString()}</span>
                    </div>
                  )}
                  {experiment.end_time && (
                    <div className="meta-item">
                      <span className="meta-label">结束时间:</span>
                      <span className="meta-value">{new Date(experiment.end_time).toLocaleString()}</span>
                    </div>
                  )}
                  {experiment.progress !== undefined && (
                    <div className="meta-item full-width">
                      <span className="meta-label">进度:</span>
                      <div className="progress-bar-container">
                        <div 
                          className="progress-bar" 
                          style={{ width: `${experiment.progress}%` }}
                        ></div>
                        <span className="progress-text">{experiment.progress.toFixed(0)}%</span>
                      </div>
                    </div>
                  )}
                </div>
                
                {experiment.performance && (
                  <div className="experiment-performance">
                    <h4>性能指标</h4>
                    <div className="performance-metrics">
                      <div className="metrics-grid">
                        {/* 核心指标 */}
                        <div className="metrics-section">
                          <h5>核心指标</h5>
                          <div className="metrics-row">
                            {experiment.performance.total_return !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">总收益</span>
                                <span className={`metric-value ${experiment.performance.total_return >= 0 ? 'positive' : 'negative'}`}>
                                  {(experiment.performance.total_return * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {experiment.performance.annual_return !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">年化收益</span>
                                <span className={`metric-value ${experiment.performance.annual_return >= 0 ? 'positive' : 'negative'}`}>
                                  {(experiment.performance.annual_return * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {experiment.performance.sharpe_ratio !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">夏普比率</span>
                                <span className={`metric-value ${experiment.performance.sharpe_ratio >= 0 ? 'positive' : 'negative'}`}>
                                  {experiment.performance.sharpe_ratio.toFixed(2)}
                                </span>
                              </div>
                            )}
                            {experiment.performance.max_drawdown !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">最大回撤</span>
                                <span className="metric-value">
                                  {(experiment.performance.max_drawdown * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        {/* 交易指标 */}
                        <div className="metrics-section">
                          <h5>交易指标</h5>
                          <div className="metrics-row">
                            {experiment.performance.win_rate !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">胜率</span>
                                <span className={`metric-value ${experiment.performance.win_rate >= 0.5 ? 'positive' : 'negative'}`}>
                                  {(experiment.performance.win_rate * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {experiment.performance.avg_win !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">平均盈利</span>
                                <span className={`metric-value ${experiment.performance.avg_win >= 0 ? 'positive' : 'negative'}`}>
                                  {(experiment.performance.avg_win * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                            {experiment.performance.avg_loss !== undefined && (
                              <div className="metric-item">
                                <span className="metric-key">平均亏损</span>
                                <span className={`metric-value ${experiment.performance.avg_loss >= 0 ? 'positive' : 'negative'}`}>
                                  {(experiment.performance.avg_loss * 100).toFixed(2)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      {/* 收益分布 */}
                      {experiment.performance.return_distribution && (
                        <div className="metrics-section">
                          <h5>收益分布</h5>
                          <div className="return-distribution">
                            <div className="distribution-item">
                              <span className="dist-key">最小值</span>
                              <span className="dist-value">{(experiment.performance.return_distribution.min * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">25%分位数</span>
                              <span className="dist-value">{(experiment.performance.return_distribution['25th'] * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">中位数</span>
                              <span className="dist-value">{(experiment.performance.return_distribution.median * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">75%分位数</span>
                              <span className="dist-value">{(experiment.performance.return_distribution['75th'] * 100).toFixed(2)}%</span>
                            </div>
                            <div className="distribution-item">
                              <span className="dist-key">最大值</span>
                              <span className="dist-value">{(experiment.performance.return_distribution.max * 100).toFixed(2)}%</span>
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
                              <ReactECharts option={getChartOption(experiment.performance, 'cumulative')} style={{ height: '100%' }} />
                            </div>
                          </div>
                          
                          {/* Drawdown Curve Chart */}
                          {experiment.performance.drawdown_curve && Object.keys(experiment.performance.drawdown_curve).length > 0 && (
                            <div className="chart-item">
                              <h6>回撤曲线</h6>
                              <div className="chart-wrapper" style={{ height: '250px' }}>
                                <ReactECharts option={getChartOption(experiment.performance, 'drawdown')} style={{ height: '100%' }} />
                              </div>
                            </div>
                          )}
                          
                          {/* Monthly Returns Chart */}
                          {experiment.performance.monthly_returns && Object.keys(experiment.performance.monthly_returns).length > 0 && (
                            <div className="chart-item">
                              <h6>月度收益</h6>
                              <div className="chart-wrapper" style={{ height: '250px' }}>
                                <ReactECharts option={getChartOption(experiment.performance, 'monthly')} style={{ height: '100%' }} />
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {experiment.error && (
                  <div className="experiment-error">
                    <h4>Error</h4>
                    <p className="error-message">{experiment.error}</p>
                  </div>
                )}
                
                <div className="experiment-actions">
                  <button 
                      className="action-btn view-btn"
                      onClick={() => navigate(`/experiments/${experiment.id}`)}
                    >
                      查看详情
                    </button>
                    {canCreateExperiments && (experiment.status === 'created' || experiment.status === 'completed' || experiment.status === 'failed') && (
                    <button 
                      className="action-btn run-btn"
                      onClick={() => handleRunExperiment(experiment.id)}
                    >
                      {experiment.status === 'created' ? '运行实验' : '重新运行'}
                    </button>
                  )}
                  <button 
                    className="action-btn logs-btn"
                    onClick={() => setShowLogs(showLogs === experiment.id ? null : experiment.id)}
                  >
                    {showLogs === experiment.id ? '隐藏日志' : '查看日志'}
                  </button>
                  {canCreateExperiments && (
                  <button 
                    className="action-btn delete-btn"
                    onClick={() => handleDeleteExperiment(experiment.id)}
                  >
                    删除
                  </button>
                  )}
                </div>
                
                {showLogs === experiment.id && (
                  <div className="experiment-logs">
                    <h4>运行日志</h4>
                    <div className="logs-container">
                      {logs[experiment.id] && logs[experiment.id].length > 0 ? (
                        <div className="logs-content">
                          {logs[experiment.id].map((log, index) => (
                            <div key={index} className="log-line">
                              {log}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="no-logs">
                          {experiment.status === 'running' ? '正在运行...' : '暂无日志'}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="experiment-list-content">
                <div className="experiment-list-header">
                  <h3 className="experiment-name">{experiment.name}</h3>
                  <div className="header-right">
                    <span className={`experiment-status status-${experiment.status}`}>
                      {experiment.status === 'created' && '已创建'}
                      {experiment.status === 'pending' && '待运行'}
                      {experiment.status === 'running' && '运行中'}
                      {experiment.status === 'completed' && '已完成'}
                      {experiment.status === 'failed' && '失败'}
                      {experiment.status === 'stopped' && '已停止'}
                      {!['created', 'pending', 'running', 'completed', 'failed', 'stopped'].includes(experiment.status) && experiment.status}
                    </span>
                    <div className="meta-item created-time">
                      <span className="meta-label">创建时间:</span>
                      <span className="meta-value">{new Date(experiment.created_at).toLocaleString()}</span>
                    </div>
                  </div>
                </div>
                <p className="experiment-description">{experiment.description}</p>
                {experiment.progress !== undefined && (
                  <div className="experiment-meta">
                    <div className="meta-item">
                      <span className="meta-label">进度:</span>
                      <span className="progress-text">{experiment.progress.toFixed(0)}%</span>
                    </div>
                  </div>
                )}
                <div className="experiment-actions">
                  <button 
                    className="action-btn view-btn"
                    onClick={() => navigate(`/experiments/${experiment.id}`)}
                  >
                    查看详情
                  </button>
                  {canCreateExperiments && (experiment.status === 'created' || experiment.status === 'completed' || experiment.status === 'failed') && (
                    <button 
                      className="action-btn run-btn"
                      onClick={() => handleRunExperiment(experiment.id)}
                    >
                      {experiment.status === 'created' ? '运行实验' : '重新运行'}
                    </button>
                  )}
                  <button 
                    className="action-btn logs-btn"
                    onClick={() => setShowLogs(showLogs === experiment.id ? null : experiment.id)}
                  >
                    {showLogs === experiment.id ? '隐藏日志' : '查看日志'}
                  </button>
                  {canCreateExperiments && (
                    <button 
                      className="action-btn delete-btn"
                      onClick={() => handleDeleteExperiment(experiment.id)}
                    >
                      删除
                    </button>
                  )}
                </div>
                {showLogs === experiment.id && (
                  <div className="experiment-logs">
                    <h4>运行日志</h4>
                    <div className="logs-container">
                      {logs[experiment.id] && logs[experiment.id].length > 0 ? (
                        <div className="logs-content">
                          {logs[experiment.id].map((log, index) => (
                            <div key={index} className="log-line">
                              {log}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="no-logs">
                          {experiment.status === 'running' ? '正在运行...' : '暂无日志'}
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

export default Experiments
