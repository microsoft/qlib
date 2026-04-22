import React, { useState, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'

interface Factor {
  id: number
  name: string
  description: string
  type: 'technical' | 'fundamental' | 'sentiment' | 'alternative'
  created_at: string
  updated_at: string
}

interface FactorAnalysisResult {
  factor_name: string
  ic_mean: number
  ic_std: number
  ir: number
  rank_ic_mean: number
  rank_ic_std: number
  rank_ir: number
  t_stat: number
  p_value: number
  sharpe_ratio: number
  total_return: number
  annual_return: number
  max_drawdown: number
}

interface CorrelationData {
  factors: string[]
  matrix: number[][]
}

interface FactorPerformance {
  date: string
  [key: string]: string | number
}

const FactorAnalysis: React.FC = () => {
  const [factors, setFactors] = useState<Factor[]>([])
  const [selectedFactors, setSelectedFactors] = useState<string[]>([])
  const [analysisResults, setAnalysisResults] = useState<FactorAnalysisResult[]>([])
  const [correlationData, setCorrelationData] = useState<CorrelationData | null>(null)
  const [factorPerformance, setFactorPerformance] = useState<FactorPerformance[]>([])
  const [loading, setLoading] = useState(true)
  const [showAnalysisForm, setShowAnalysisForm] = useState(false)
  const [analysisType, setAnalysisType] = useState<'ic_ir' | 'correlation' | 'performance'>('ic_ir')

  // 模拟因子数据
  useEffect(() => {
    const mockFactors: Factor[] = [
      {
        id: 1,
        name: 'Alpha158',
        description: '经典量化因子组合',
        type: 'technical',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 2,
        name: '动量因子',
        description: '基于价格动量的因子',
        type: 'technical',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 3,
        name: '反转因子',
        description: '基于价格反转的因子',
        type: 'technical',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 4,
        name: '波动率因子',
        description: '基于价格波动率的因子',
        type: 'technical',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 5,
        name: '换手率因子',
        description: '基于成交量换手率的因子',
        type: 'technical',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 6,
        name: 'PE因子',
        description: '基于市盈率的因子',
        type: 'fundamental',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 7,
        name: 'PB因子',
        description: '基于市净率的因子',
        type: 'fundamental',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ]

    const mockAnalysisResults: FactorAnalysisResult[] = [
      {
        factor_name: 'Alpha158',
        ic_mean: 0.035,
        ic_std: 0.082,
        ir: 0.427,
        rank_ic_mean: 0.032,
        rank_ic_std: 0.078,
        rank_ir: 0.410,
        t_stat: 3.85,
        p_value: 0.001,
        sharpe_ratio: 1.85,
        total_return: 0.452,
        annual_return: 0.128,
        max_drawdown: -0.185
      },
      {
        factor_name: '动量因子',
        ic_mean: 0.028,
        ic_std: 0.075,
        ir: 0.373,
        rank_ic_mean: 0.026,
        rank_ic_std: 0.072,
        rank_ir: 0.361,
        t_stat: 3.42,
        p_value: 0.002,
        sharpe_ratio: 1.62,
        total_return: 0.385,
        annual_return: 0.112,
        max_drawdown: -0.213
      },
      {
        factor_name: '反转因子',
        ic_mean: 0.022,
        ic_std: 0.068,
        ir: 0.324,
        rank_ic_mean: 0.020,
        rank_ic_std: 0.065,
        rank_ir: 0.308,
        t_stat: 2.95,
        p_value: 0.005,
        sharpe_ratio: 1.45,
        total_return: 0.328,
        annual_return: 0.095,
        max_drawdown: -0.237
      },
      {
        factor_name: '波动率因子',
        ic_mean: 0.018,
        ic_std: 0.062,
        ir: 0.290,
        rank_ic_mean: 0.016,
        rank_ic_std: 0.059,
        rank_ir: 0.271,
        t_stat: 2.68,
        p_value: 0.009,
        sharpe_ratio: 1.28,
        total_return: 0.285,
        annual_return: 0.082,
        max_drawdown: -0.256
      }
    ]

    const mockCorrelationData: CorrelationData = {
      factors: ['Alpha158', '动量因子', '反转因子', '波动率因子', '换手率因子', 'PE因子', 'PB因子'],
      matrix: [
        [1.00, 0.35, -0.28, 0.15, 0.22, -0.18, -0.15],
        [0.35, 1.00, -0.42, 0.28, 0.35, -0.22, -0.19],
        [-0.28, -0.42, 1.00, -0.15, -0.20, 0.18, 0.15],
        [0.15, 0.28, -0.15, 1.00, 0.45, -0.32, -0.28],
        [0.22, 0.35, -0.20, 0.45, 1.00, -0.25, -0.22],
        [-0.18, -0.22, 0.18, -0.32, -0.25, 1.00, 0.85],
        [-0.15, -0.19, 0.15, -0.28, -0.22, 0.85, 1.00]
      ]
    }

    const mockFactorPerformance: FactorPerformance[] = Array.from({ length: 24 }, (_, i) => {
      const date = new Date()
      date.setMonth(date.getMonth() - i)
      return {
        date: date.toISOString().split('T')[0],
        Alpha158: 0.01 + Math.random() * 0.05 - 0.025,
        '动量因子': 0.008 + Math.random() * 0.04 - 0.02,
        '反转因子': 0.006 + Math.random() * 0.035 - 0.0175,
        '波动率因子': 0.005 + Math.random() * 0.03 - 0.015
      }
    }).reverse()

    setFactors(mockFactors)
    setSelectedFactors(['Alpha158', '动量因子', '反转因子', '波动率因子'])
    setAnalysisResults(mockAnalysisResults)
    setCorrelationData(mockCorrelationData)
    setFactorPerformance(mockFactorPerformance)
    setLoading(false)
  }, [])

  // 生成IC/IR分析图表
  const getICIRChartOption = () => {
    return {
      title: {
        text: '因子IC/IR分析',
        left: 'center',
        textStyle: {
          fontSize: 14
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        formatter: (params: any) => {
          let result = `${params[0].axisValue}<br/>`
          params.forEach((param: any) => {
            result += `${param.seriesName}: ${param.value.toFixed(4)}<br/>`
          })
          return result
        }
      },
      legend: {
        data: ['IC均值', 'IR', 'Rank IC均值', 'Rank IR'],
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
        data: analysisResults.map(result => result.factor_name),
        axisLabel: {
          rotate: 45,
          fontSize: 11
        }
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          fontSize: 12
        }
      },
      series: [
        {
          name: 'IC均值',
          type: 'bar',
          data: analysisResults.map(result => result.ic_mean),
          itemStyle: {
            color: '#646cff'
          }
        },
        {
          name: 'IR',
          type: 'bar',
          data: analysisResults.map(result => result.ir),
          itemStyle: {
            color: '#81c784'
          }
        },
        {
          name: 'Rank IC均值',
          type: 'bar',
          data: analysisResults.map(result => result.rank_ic_mean),
          itemStyle: {
            color: '#ffb74d'
          }
        },
        {
          name: 'Rank IR',
          type: 'bar',
          data: analysisResults.map(result => result.rank_ir),
          itemStyle: {
            color: '#e57373'
          }
        }
      ]
    }
  }

  // 生成因子相关性热力图
  const getCorrelationHeatmapOption = () => {
    if (!correlationData) {
      return {}
    }

    return {
      title: {
        text: '因子相关性矩阵',
        left: 'center',
        textStyle: {
          fontSize: 14
        }
      },
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const { dataIndex, name, value } = params
          const xFactor = correlationData.factors[dataIndex]
          const yFactor = name
          return `${xFactor} vs ${yFactor}: ${value.toFixed(4)}`
        }
      },
      grid: {
        height: '50%',
        top: '15%',
        bottom: '25%'
      },
      xAxis: {
        type: 'category',
        data: correlationData.factors,
        splitArea: {
          show: true
        },
        axisLabel: {
          rotate: 45,
          fontSize: 11
        }
      },
      yAxis: {
        type: 'category',
        data: correlationData.factors,
        splitArea: {
          show: true
        },
        axisLabel: {
          fontSize: 11
        }
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        textStyle: {
          fontSize: 12
        },
        inRange: {
          color: ['#64b5f6', '#ffffff', '#ef5350']
        }
      },
      series: [
        {
          name: '相关性',
          type: 'heatmap',
          data: correlationData.matrix.flatMap((row, i) => 
            row.map((value, j) => [i, j, value])
          ),
          label: {
            show: true,
            fontSize: 10,
            formatter: (params: any) => params.value.toFixed(2)
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    }
  }

  // 生成因子表现对比图
  const getFactorPerformanceChartOption = () => {
    const factorNames = Object.keys(factorPerformance[0]).filter(key => key !== 'date')
    
    return {
      title: {
        text: '因子表现对比',
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
            result += `${param.seriesName}: ${(param.value * 100).toFixed(2)}%<br/>`
          })
          return result
        }
      },
      legend: {
        data: factorNames,
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
        data: factorPerformance.map(item => item.date),
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
      series: factorNames.map(name => ({
        name,
        type: 'line',
        data: factorPerformance.map(item => (item[name] as number) * 100),
        smooth: true,
        lineStyle: {
          width: 2
        },
        areaStyle: {
          opacity: 0.1
        }
      }))
    }
  }

  // 生成因子有效性表格
  const renderAnalysisResultsTable = () => {
    return (
      <div className="analysis-results-table">
        <table className="results-table">
          <thead>
            <tr>
              <th>因子名称</th>
              <th>IC均值</th>
              <th>IC标准差</th>
              <th>IR</th>
              <th>Rank IC均值</th>
              <th>Rank IC标准差</th>
              <th>Rank IR</th>
              <th>T统计量</th>
              <th>P值</th>
              <th>夏普比率</th>
            </tr>
          </thead>
          <tbody>
            {analysisResults.map((result, index) => (
              <tr key={index}>
                <td>{result.factor_name}</td>
                <td className={result.ic_mean > 0 ? 'positive' : 'negative'}>
                  {result.ic_mean.toFixed(4)}
                </td>
                <td>{result.ic_std.toFixed(4)}</td>
                <td className={result.ir > 0.3 ? 'positive' : result.ir > 0 ? 'warning' : 'negative'}>
                  {result.ir.toFixed(4)}
                </td>
                <td className={result.rank_ic_mean > 0 ? 'positive' : 'negative'}>
                  {result.rank_ic_mean.toFixed(4)}
                </td>
                <td>{result.rank_ic_std.toFixed(4)}</td>
                <td className={result.rank_ir > 0.3 ? 'positive' : result.rank_ir > 0 ? 'warning' : 'negative'}>
                  {result.rank_ir.toFixed(4)}
                </td>
                <td>{result.t_stat.toFixed(2)}</td>
                <td>{result.p_value.toFixed(4)}</td>
                <td className={result.sharpe_ratio > 1 ? 'positive' : result.sharpe_ratio > 0 ? 'warning' : 'negative'}>
                  {result.sharpe_ratio.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  if (loading) {
    return <div className="container">Loading...</div>
  }

  return (
    <div className="container page-transition">
      <div className="page-header">
        <h1>因子分析</h1>
        <div className="header-actions">
          <button className="btn" onClick={() => setShowAnalysisForm(true)}>
            运行因子分析
          </button>
        </div>
      </div>

      {/* 因子选择和分析类型 */}
      <div className="analysis-controls">
        <div className="control-group">
          <label>选择因子</label>
          <div className="factor-selector">
            {factors.map(factor => (
              <label key={factor.id} className="factor-checkbox">
                <input
                  type="checkbox"
                  value={factor.name}
                  checked={selectedFactors.includes(factor.name)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedFactors([...selectedFactors, factor.name])
                    } else {
                      setSelectedFactors(selectedFactors.filter(name => name !== factor.name))
                    }
                  }}
                />
                <span>{factor.name}</span>
              </label>
            ))}
          </div>
        </div>
        <div className="control-group">
          <label>分析类型</label>
          <div className="analysis-type-selector">
            <button
              className={`analysis-type-btn ${analysisType === 'ic_ir' ? 'active' : ''}`}
              onClick={() => setAnalysisType('ic_ir')}
            >
              IC/IR分析
            </button>
            <button
              className={`analysis-type-btn ${analysisType === 'correlation' ? 'active' : ''}`}
              onClick={() => setAnalysisType('correlation')}
            >
              相关性分析
            </button>
            <button
              className={`analysis-type-btn ${analysisType === 'performance' ? 'active' : ''}`}
              onClick={() => setAnalysisType('performance')}
            >
              表现对比
            </button>
          </div>
        </div>
      </div>

      {/* 分析结果展示 */}
      <div className="analysis-results">
        {analysisType === 'ic_ir' && (
          <>
            <h2 className="section-title">IC/IR分析结果</h2>
            <div className="chart-container">
              <ReactECharts option={getICIRChartOption()} style={{ height: '400px' }} />
            </div>
            {renderAnalysisResultsTable()}
          </>
        )}

        {analysisType === 'correlation' && (
          <>
            <h2 className="section-title">因子相关性分析</h2>
            <div className="chart-container">
              <ReactECharts option={getCorrelationHeatmapOption()} style={{ height: '500px' }} />
            </div>
            <div className="correlation-legend">
              <h3>相关性解释</h3>
              <ul>
                <li><span className="color-box positive"></span> 强正相关 (0.7-1.0)</li>
                <li><span className="color-box weak-positive"></span> 弱正相关 (0.3-0.7)</li>
                <li><span className="color-box neutral"></span> 中性 (-0.3-0.3)</li>
                <li><span className="color-box weak-negative"></span> 弱负相关 (-0.7--0.3)</li>
                <li><span className="color-box negative"></span> 强负相关 (-1.0--0.7)</li>
              </ul>
            </div>
          </>
        )}

        {analysisType === 'performance' && (
          <>
            <h2 className="section-title">因子表现对比</h2>
            <div className="chart-container">
              <ReactECharts option={getFactorPerformanceChartOption()} style={{ height: '400px' }} />
            </div>
            <div className="performance-summary">
              <h3>表现总结</h3>
              <div className="summary-stats">
                {analysisResults.map((result, index) => (
                  <div key={index} className="summary-card">
                    <h4>{result.factor_name}</h4>
                    <div className="summary-metrics">
                      <div className="metric-item">
                        <span className="metric-label">总收益</span>
                        <span className={`metric-value ${result.total_return >= 0 ? 'positive' : 'negative'}`}>
                          {(result.total_return * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">年化收益</span>
                        <span className={`metric-value ${result.annual_return >= 0 ? 'positive' : 'negative'}`}>
                          {(result.annual_return * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">最大回撤</span>
                        <span className="metric-value">
                          {(result.max_drawdown * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* 因子分析设置表单 */}
      {showAnalysisForm && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h2>运行因子分析</h2>
              <button className="modal-close" onClick={() => setShowAnalysisForm(false)}>×</button>
            </div>
            <div className="modal-body">
              <form>
                <div className="form-group">
                  <label>选择分析周期</label>
                  <select className="form-control">
                    <option value="monthly">月度</option>
                    <option value="weekly">周度</option>
                    <option value="daily">日度</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>开始日期</label>
                  <input type="date" className="form-control" />
                </div>
                <div className="form-group">
                  <label>结束日期</label>
                  <input type="date" className="form-control" />
                </div>
                <div className="form-group">
                  <label>显著性水平</label>
                  <input type="number" className="form-control" placeholder="0.05" step="0.01" />
                </div>
                <div className="form-actions">
                  <button type="submit" className="btn">开始分析</button>
                  <button type="button" className="btn" onClick={() => setShowAnalysisForm(false)}>取消</button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default FactorAnalysis
