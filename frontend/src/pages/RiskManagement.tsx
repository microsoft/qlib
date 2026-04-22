import React, { useState, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'

interface RiskIndicator {
  name: string
  current_value: number
  threshold: number
  unit: string
  status: 'normal' | 'warning' | 'danger'
  description: string
  trend: 'up' | 'down' | 'stable'
  historical_data: { date: string; value: number }[]
}

interface RiskAlert {
  id: number
  timestamp: string
  indicator: string
  level: 'info' | 'warning' | 'critical'
  message: string
  resolved: boolean
}

interface StopLossStrategy {
  id: number
  name: string
  description: string
  type: 'fixed' | 'trailing' | 'volatility'
  parameters: any
  enabled: boolean
}

const RiskManagement: React.FC = () => {
  const [riskIndicators, setRiskIndicators] = useState<RiskIndicator[]>([])
  const [riskAlerts, setRiskAlerts] = useState<RiskAlert[]>([])
  const [stopLossStrategies, setStopLossStrategies] = useState<StopLossStrategy[]>([])
  const [loading, setLoading] = useState(true)
  const [showAlertForm, setShowAlertForm] = useState(false)
  const [showStopLossForm, setShowStopLossForm] = useState(false)

  // 模拟风险指标数据
  useEffect(() => {
    const mockRiskIndicators: RiskIndicator[] = [
      {
        name: '风险价值 (VaR)',
        current_value: 4.2,
        threshold: 5.0,
        unit: '%',
        status: 'normal',
        description: '95%置信区间下的日最大可能损失',
        trend: 'up',
        historical_data: Array.from({ length: 30 }, (_, i) => {
          const date = new Date()
          date.setDate(date.getDate() - i)
          return {
            date: date.toISOString().split('T')[0],
            value: 3.5 + Math.random() * 1.5
          }
        }).reverse()
      },
      {
        name: '夏普比率',
        current_value: 1.85,
        threshold: 1.0,
        unit: '',
        status: 'normal',
        description: '风险调整后收益率',
        trend: 'stable',
        historical_data: Array.from({ length: 30 }, (_, i) => {
          const date = new Date()
          date.setDate(date.getDate() - i)
          return {
            date: date.toISOString().split('T')[0],
            value: 1.7 + Math.random() * 0.4
          }
        }).reverse()
      },
      {
        name: '最大回撤',
        current_value: -12.3,
        threshold: -20.0,
        unit: '%',
        status: 'normal',
        description: '历史最大跌幅',
        trend: 'down',
        historical_data: Array.from({ length: 30 }, (_, i) => {
          const date = new Date()
          date.setDate(date.getDate() - i)
          return {
            date: date.toISOString().split('T')[0],
            value: -10.0 - Math.random() * 5.0
          }
        }).reverse()
      },
      {
        name: '波动率',
        current_value: 15.6,
        threshold: 20.0,
        unit: '%',
        status: 'normal',
        description: '年化波动率',
        trend: 'up',
        historical_data: Array.from({ length: 30 }, (_, i) => {
          const date = new Date()
          date.setDate(date.getDate() - i)
          return {
            date: date.toISOString().split('T')[0],
            value: 14.0 + Math.random() * 3.0
          }
        }).reverse()
      },
      {
        name: '贝塔系数',
        current_value: 0.85,
        threshold: 1.2,
        unit: '',
        status: 'normal',
        description: '与市场相关性',
        trend: 'stable',
        historical_data: Array.from({ length: 30 }, (_, i) => {
          const date = new Date()
          date.setDate(date.getDate() - i)
          return {
            date: date.toISOString().split('T')[0],
            value: 0.8 + Math.random() * 0.2
          }
        }).reverse()
      },
      {
        name: '换手率',
        current_value: 25.3,
        threshold: 30.0,
        unit: '%',
        status: 'warning',
        description: '月换手率',
        trend: 'up',
        historical_data: Array.from({ length: 30 }, (_, i) => {
          const date = new Date()
          date.setDate(date.getDate() - i)
          return {
            date: date.toISOString().split('T')[0],
            value: 20.0 + Math.random() * 8.0
          }
        }).reverse()
      }
    ]

    const mockRiskAlerts: RiskAlert[] = [
      {
        id: 1,
        timestamp: new Date().toISOString().split('T')[0] + ' 14:30:00',
        indicator: '换手率',
        level: 'warning',
        message: '换手率接近阈值 (当前: 25.3%, 阈值: 30.0%)',
        resolved: false
      },
      {
        id: 2,
        timestamp: new Date().toISOString().split('T')[0] + ' 10:15:00',
        indicator: '波动率',
        level: 'info',
        message: '波动率上升趋势明显',
        resolved: true
      },
      {
        id: 3,
        timestamp: new Date(Date.now() - 86400000).toISOString().split('T')[0] + ' 16:45:00',
        indicator: '最大回撤',
        level: 'critical',
        message: '最大回撤达到 -12.3%',
        resolved: true
      }
    ]

    const mockStopLossStrategies: StopLossStrategy[] = [
      {
        id: 1,
        name: '固定止损策略',
        description: '单个资产跌幅超过5%时止损',
        type: 'fixed',
        parameters: {
          threshold: 5.0
        },
        enabled: true
      },
      {
        id: 2,
        name: '跟踪止损策略',
        description: '盈利回吐10%时止损',
        type: 'trailing',
        parameters: {
          trail_percent: 10.0
        },
        enabled: true
      },
      {
        id: 3,
        name: '波动率止损策略',
        description: '基于ATR的动态止损',
        type: 'volatility',
        parameters: {
          atr_multiplier: 2.0
        },
        enabled: false
      }
    ]

    setRiskIndicators(mockRiskIndicators)
    setRiskAlerts(mockRiskAlerts)
    setStopLossStrategies(mockStopLossStrategies)
    setLoading(false)
  }, [])

  // 生成指标趋势图配置
  const getIndicatorChartOption = (indicator: RiskIndicator) => {
    return {
      title: {
        text: indicator.name,
        left: 'center',
        textStyle: {
          fontSize: 14,
          fontWeight: 'normal'
        }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const date = params[0].axisValue
          const value = params[0].value
          return `${date}<br/>${indicator.name}: ${value}${indicator.unit}`
        }
      },
      xAxis: {
        type: 'category',
        data: indicator.historical_data.map(item => item.date),
        axisLabel: {
          rotate: 45,
          fontSize: 10
        }
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: `{value}${indicator.unit}`,
          fontSize: 10
        },
        splitLine: {
          lineStyle: {
            type: 'dashed'
          }
        }
      },
      series: [
        {
          data: indicator.historical_data.map(item => item.value),
          type: 'line',
          smooth: true,
          itemStyle: {
            color: '#646cff'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: '#646cff80' },
                { offset: 1, color: '#646cff10' }
              ]
            }
          }
        }
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        top: '15%',
        containLabel: true
      }
    }
  }

  if (loading) {
    return <div className="container">Loading...</div>
  }

  return (
    <div className="container page-transition">
      <div className="page-header">
        <h1>风险控制</h1>
        <div className="header-actions">
          <button className="btn" onClick={() => setShowAlertForm(true)}>
            设置风险预警
          </button>
          <button className="btn" onClick={() => setShowStopLossForm(true)}>
            管理止损策略
          </button>
        </div>
      </div>

      {/* 风险指标概览 */}
      <div className="risk-metrics-section">
        <h2 className="section-title">风险指标实时监控</h2>
        <div className="risk-metrics-grid">
          {riskIndicators.map((indicator, index) => (
            <div key={index} className={`risk-indicator-card status-${indicator.status}`}>
              <div className="indicator-header">
                <div className="indicator-info">
                  <h3 className="indicator-name">{indicator.name}</h3>
                  <p className="indicator-description">{indicator.description}</p>
                </div>
                <div className={`indicator-trend trend-${indicator.trend}`}>
                  {indicator.trend === 'up' ? '↑' : indicator.trend === 'down' ? '↓' : '→'}
                </div>
              </div>
              <div className="indicator-value-container">
                <div className="indicator-value">
                  {indicator.current_value}{indicator.unit}
                </div>
                <div className="indicator-threshold">
                  阈值: {indicator.threshold}{indicator.unit}
                </div>
              </div>
              <div className="indicator-chart">
                <ReactECharts option={getIndicatorChartOption(indicator)} style={{ height: '150px' }} />
              </div>
              <div className="indicator-status">
                <span className={`status-badge status-${indicator.status}`}>
                  {indicator.status === 'normal' && '正常'}
                  {indicator.status === 'warning' && '警告'}
                  {indicator.status === 'danger' && '危险'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 风险预警 */}
      <div className="risk-alerts-section">
        <h2 className="section-title">风险预警</h2>
        <div className="alerts-container">
          {riskAlerts.map(alert => (
            <div key={alert.id} className={`alert-item level-${alert.level} ${alert.resolved ? 'resolved' : ''}`}>
              <div className="alert-header">
                <div className="alert-meta">
                  <span className={`alert-level level-${alert.level}`}>
                    {alert.level === 'info' && '💡'}
                    {alert.level === 'warning' && '⚠️'}
                    {alert.level === 'critical' && '🚨'}
                  </span>
                  <span className="alert-timestamp">{alert.timestamp}</span>
                </div>
                <div className="alert-actions">
                  {!alert.resolved && (
                    <button 
                      className="btn btn-sm btn-secondary"
                      onClick={() => {
                        setRiskAlerts(alerts => alerts.map(a => 
                          a.id === alert.id ? { ...a, resolved: true } : a
                        ))
                      }}
                    >
                      标记已处理
                    </button>
                  )}
                </div>
              </div>
              <div className="alert-content">
                <h4 className="alert-indicator">{alert.indicator}</h4>
                <p className="alert-message">{alert.message}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 止损策略 */}
      <div className="stop-loss-section">
        <h2 className="section-title">止损策略管理</h2>
        <div className="stop-loss-grid">
          {stopLossStrategies.map(strategy => (
            <div key={strategy.id} className={`stop-loss-card ${strategy.enabled ? 'enabled' : 'disabled'}`}>
              <div className="strategy-header">
                <h3 className="strategy-name">{strategy.name}</h3>
                <div className={`strategy-status ${strategy.enabled ? 'enabled' : 'disabled'}`}>
                  {strategy.enabled ? '已启用' : '已禁用'}
                </div>
              </div>
              <p className="strategy-description">{strategy.description}</p>
              <div className="strategy-details">
                <div className="strategy-type">
                  类型: {strategy.type === 'fixed' ? '固定止损' : 
                       strategy.type === 'trailing' ? '跟踪止损' : '波动率止损'}
                </div>
                <div className="strategy-parameters">
                  参数: {JSON.stringify(strategy.parameters)}
                </div>
              </div>
              <div className="strategy-actions">
                <button 
                  className={`btn btn-sm ${strategy.enabled ? 'btn-secondary' : 'btn-primary'}`}
                  onClick={() => {
                    setStopLossStrategies(strategies => strategies.map(s => 
                      s.id === strategy.id ? { ...s, enabled: !s.enabled } : s
                    ))
                  }}
                >
                  {strategy.enabled ? '禁用' : '启用'}
                </button>
                <button className="btn btn-sm btn-secondary">
                  编辑
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 风险控制面板 */}
      <div className="risk-dashboard-section">
        <h2 className="section-title">风险控制面板</h2>
        <div className="dashboard-grid">
          <div className="dashboard-card">
            <h3>风险暴露分析</h3>
            <div className="exposure-chart">
              <ReactECharts 
                option={{
                  title: {
                    text: '资产类别暴露',
                    left: 'center',
                    textStyle: {
                      fontSize: 13
                    }
                  },
                  tooltip: {
                    trigger: 'item',
                    formatter: '{b}: {c} ({d}%)'
                  },
                  legend: {
                    orient: 'horizontal',
                    bottom: 10,
                    textStyle: {
                      fontSize: 12
                    }
                  },
                  series: [
                    {
                      name: '资产类别',
                      type: 'pie',
                      radius: ['50%', '70%'],
                      center: ['50%', '45%'],
                      data: [
                        { value: 45, name: '股票' },
                        { value: 30, name: '债券' },
                        { value: 15, name: '商品' },
                        { value: 10, name: '现金' }
                      ],
                      emphasis: {
                        itemStyle: {
                          shadowBlur: 10,
                          shadowOffsetX: 0,
                          shadowColor: 'rgba(0, 0, 0, 0.5)'
                        }
                      }
                    }
                  ]
                }} 
                style={{ height: '250px' }} 
              />
            </div>
          </div>
          
          <div className="dashboard-card">
            <h3>风险贡献度</h3>
            <div className="contribution-chart">
              <ReactECharts 
                option={{
                  title: {
                    text: '风险贡献度',
                    left: 'center',
                    textStyle: {
                      fontSize: 13
                    }
                  },
                  tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                      type: 'shadow'
                    }
                  },
                  xAxis: {
                    type: 'category',
                    data: ['VaR', '夏普比率', '最大回撤', '波动率', '贝塔系数', '换手率'],
                    axisLabel: {
                      rotate: 45,
                      fontSize: 11
                    }
                  },
                  yAxis: {
                    type: 'value',
                    name: '贡献度 (%)'
                  },
                  series: [
                    {
                      data: [25, 20, 18, 15, 12, 10],
                      type: 'bar',
                      itemStyle: {
                        color: '#646cff'
                      }
                    }
                  ]
                }} 
                style={{ height: '250px' }} 
              />
            </div>
          </div>
        </div>
      </div>

      {/* 风险预警设置表单 */}
      {showAlertForm && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h2>设置风险预警</h2>
              <button className="modal-close" onClick={() => setShowAlertForm(false)}>×</button>
            </div>
            <div className="modal-body">
              <form>
                <div className="form-group">
                  <label>选择指标</label>
                  <select className="form-control">
                    {riskIndicators.map((indicator, index) => (
                      <option key={index}>{indicator.name}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>预警阈值</label>
                  <input type="number" className="form-control" placeholder="输入预警阈值" />
                </div>
                <div className="form-group">
                  <label>预警级别</label>
                  <select className="form-control">
                    <option value="info">信息</option>
                    <option value="warning">警告</option>
                    <option value="critical">严重</option>
                  </select>
                </div>
                <div className="form-actions">
                  <button type="submit" className="btn">保存设置</button>
                  <button type="button" className="btn" onClick={() => setShowAlertForm(false)}>取消</button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* 止损策略管理表单 */}
      {showStopLossForm && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h2>管理止损策略</h2>
              <button className="modal-close" onClick={() => setShowStopLossForm(false)}>×</button>
            </div>
            <div className="modal-body">
              <form>
                <div className="form-group">
                  <label>策略名称</label>
                  <input type="text" className="form-control" placeholder="输入策略名称" />
                </div>
                <div className="form-group">
                  <label>策略描述</label>
                  <textarea className="form-control" rows={3} placeholder="输入策略描述"></textarea>
                </div>
                <div className="form-group">
                  <label>策略类型</label>
                  <select className="form-control">
                    <option value="fixed">固定止损</option>
                    <option value="trailing">跟踪止损</option>
                    <option value="volatility">波动率止损</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>参数设置</label>
                  <input type="text" className="form-control" placeholder="输入参数 JSON" />
                </div>
                <div className="form-actions">
                  <button type="submit" className="btn">保存策略</button>
                  <button type="button" className="btn" onClick={() => setShowStopLossForm(false)}>取消</button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default RiskManagement
