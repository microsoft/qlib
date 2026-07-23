import React, { useState, useEffect } from 'react'
import { getStockData, getStockCodes, alignData, getInstruments } from '../services/data'
import Select from 'react-select'
import './DataManagement.css'
import {
  LineChart,
  CandlestickChart,
  BarChart,
  MultiStockChart,
  TechnicalIndicatorsChart,
  HeatmapChart,
  DistributionChart
} from '../components/Charts'

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

interface DataResponse {
  data: StockData[]
  total: number
  page: number
  per_page: number
}

const DataManagement: React.FC = () => {
  const [stockData, setStockData] = useState<StockData[]>([])
  const [stockCodes, setStockCodes] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [filtering, setFiltering] = useState(false)
  const [stockCode, setStockCode] = useState('')
  const [market, setMarket] = useState('all')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [nameFilter, setNameFilter] = useState('')
  const [expressionFilter, setExpressionFilter] = useState('')
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(100)
  const [total, setTotal] = useState(0)
  const [error, setError] = useState('')
  // 数据对齐相关状态
  const [alignMode, setAlignMode] = useState<'auto' | 'manual'>('auto')
  const [alignDate, setAlignDate] = useState<string>(new Date().toISOString().split('T')[0])
  const [aligning, setAligning] = useState(false)
  const [alignStatus, setAlignStatus] = useState<string>('')
  // 特征选择相关状态
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(['open', 'high', 'low', 'close', 'volume'])
  const [availableFeatures, setAvailableFeatures] = useState<{ name: string; description: string }[]>([
    { name: 'open', description: '开盘价' },
    { name: 'high', description: '最高价' },
    { name: 'low', description: '最低价' },
    { name: 'close', description: '收盘价' },
    { name: 'volume', description: '成交量' }
  ])
  const [showFeatureSelector, setShowFeatureSelector] = useState(false)
  const [customFeature, setCustomFeature] = useState('')
  const [customFeatureName, setCustomFeatureName] = useState('')
  // 图表相关状态
  const [showChart, setShowChart] = useState(false)
  const [chartType, setChartType] = useState<'line' | 'candlestick' | 'bar' | 'multi' | 'technical' | 'heatmap' | 'distribution'>('line')
  const [chartStock, setChartStock] = useState('')
  const [chartStocks, setChartStocks] = useState<string[]>([])
  const [chartFeature, setChartFeature] = useState('close')
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['MA5', 'MA10', 'MA20'])
  const [distributionChartType, setDistributionChartType] = useState<'histogram' | 'boxplot'>('histogram')

  useEffect(() => {
    fetchStockCodes()
  }, [market, nameFilter])

  useEffect(() => {
    fetchStockData()
  }, [stockCode, startDate, endDate, page, perPage])

  const fetchStockCodes = async () => {
    try {
      // Get stock codes with filters
      const filters = {
        market: market !== 'all' ? market : undefined,
        name_filter: nameFilter || undefined
      }
      const codes = await getInstruments(filters)
      setStockCodes(codes)
    } catch (err) {
      console.error('Error fetching stock codes:', err)
      // Fallback to original method if new endpoint fails
      try {
        const codes = await getStockCodes()
        setStockCodes(codes)
      } catch (fallbackErr) {
        console.error('Fallback error fetching stock codes:', fallbackErr)
      }
    }
  }

  const fetchStockData = async () => {
    try {
      setLoading(true)
      const response: DataResponse = await getStockData({
        stock_code: stockCode || undefined,
        market: market !== 'all' ? market : undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        page,
        per_page: perPage
      })
      setStockData(response.data)
      setTotal(response.total)
    } catch (err) {
      setError('Failed to fetch stock data')
      console.error('Error fetching stock data:', err)
    } finally {
      setLoading(false)
      setFiltering(false)
    }
  }

  const handleFilter = () => {
    setPage(1)
    setFiltering(true)
  }

  const handleReset = () => {
    setStockCode('')
    setMarket('all')
    setStartDate('')
    setEndDate('')
    setNameFilter('')
    setExpressionFilter('')
    setPage(1)
    setFiltering(true)
  }

  const handlePageChange = (newPage: number) => {
    setPage(newPage)
  }

  const handlePerPageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setPerPage(parseInt(e.target.value))
    setPage(1)
  }

  // 特征选择相关函数
  const toggleFeature = (feature: string) => {
    setSelectedFeatures(prev => {
      if (prev.includes(feature)) {
        // 确保至少保留一个特征
        if (prev.length <= 1) {
          return prev
        }
        return prev.filter(f => f !== feature)
      } else {
        return [...prev, feature]
      }
    })
  }

  const addCustomFeature = () => {
    if (customFeature && customFeatureName) {
      const newFeature = {
        name: customFeatureName,
        description: customFeature
      }
      setAvailableFeatures(prev => [...prev, newFeature])
      setSelectedFeatures(prev => [...prev, customFeatureName])
      setCustomFeature('')
      setCustomFeatureName('')
    }
  }

  // 图表相关函数
  const toggleStockSelection = (stockCode: string) => {
    setChartStocks(prev => {
      if (prev.includes(stockCode)) {
        return prev.filter(code => code !== stockCode);
      } else {
        // Limit to 5 stocks for performance
        if (prev.length >= 5) {
          return prev;
        }
        return [...prev, stockCode];
      }
    });
  };

  const toggleIndicator = (indicator: string) => {
    setSelectedIndicators(prev => {
      if (prev.includes(indicator)) {
        return prev.filter(i => i !== indicator);
      } else {
        return [...prev, indicator];
      }
    });
  };

  // 导出数据为CSV
  const exportToCSV = () => {
    if (stockData.length === 0) {
      return
    }

    // Define CSV headers
    const headers = ['股票代码', '日期', ...selectedFeatures]
    
    // Convert data to CSV rows
    const rows = stockData.map(data => {
      const row = [
        data.stock_code,
        data.date,
        ...selectedFeatures.map(feature => data[feature])
      ]
      return row.map(cell => {
        // Handle commas in cell values
        if (typeof cell === 'string' && cell.includes(',')) {
          return `"${cell}"`
        }
        return cell
      }).join(',')
    })
    
    // Combine headers and rows
    const csvContent = [headers.join(','), ...rows].join('\n')
    
    // Create a blob and download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    const url = URL.createObjectURL(blob)
    link.setAttribute('href', url)
    link.setAttribute('download', `stock_data_${new Date().toISOString().split('T')[0]}.csv`)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const totalPages = Math.ceil(total / perPage)

  // 处理数据对齐
  const handleAlignData = async () => {
    try {
      setAligning(true)
      setAlignStatus('正在执行数据对齐...')
      await alignData({ mode: alignMode, date: alignDate })
      setAlignStatus('数据对齐完成，正在刷新数据...')
      // 数据对齐完成后自动刷新数据列表
      await fetchStockData()
      setAlignStatus('数据对齐完成，数据已刷新')
    } catch (err) {
      console.error('数据对齐失败:', err)
      setAlignStatus('数据对齐失败')
    } finally {
      setAligning(false)
    }
  }

  return (
    <div className="container">
      <div className="page-header">
        <h1>数据管理</h1>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {/* 数据对齐配置区域 */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <h2 style={{ marginBottom: '15px', fontSize: '1.2rem' }}>数据对齐配置</h2>
        <div className="align-config">
          <div className="align-mode">
            <label>对齐模式：</label>
            <div className="radio-group">
              <label className="radio-item">
                <input 
                  type="radio" 
                  value="auto" 
                  checked={alignMode === 'auto'} 
                  onChange={(e) => setAlignMode(e.target.value as 'auto' | 'manual')}
                />
                自动
              </label>
              <label className="radio-item">
                <input 
                  type="radio" 
                  value="manual" 
                  checked={alignMode === 'manual'} 
                  onChange={(e) => setAlignMode(e.target.value as 'auto' | 'manual')}
                />
                手动
              </label>
            </div>
          </div>
          
          <div className="align-date">
            <label htmlFor="alignDate">对齐日期：</label>
            <input
              type="date"
              id="alignDate"
              value={alignDate}
              onChange={(e) => setAlignDate(e.target.value)}
              className="form-control"
            />
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={handleAlignData}
            disabled={aligning}
          >
            {aligning ? '对齐中...' : '手动对齐数据'}
          </button>
          
          {alignStatus && (
            <div className={`align-status ${alignStatus.includes('失败') ? 'status-error' : 'status-success'}`}>
              {alignStatus}
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <h2>数据筛选</h2>
        <div className="filter-form">
          <div className="filter-group">
            <label htmlFor="market">市场</label>
            <select
              id="market"
              value={market}
              onChange={(e) => setMarket(e.target.value)}
              className="form-control"
            >
              <option value="all">全部</option>
              <option value="csi300">CSI300</option>
              <option value="csi500">CSI500</option>
              <option value="csi800">CSI800</option>
              <option value="csi1000">CSI1000</option>
            </select>
          </div>
          <div className="filter-group">
            <label htmlFor="stockCode">股票代码</label>
            <Select
              id="stockCode"
              value={stockCode ? { value: stockCode, label: stockCode } : null}
              onChange={(selectedOption: { value: string; label: string } | null) => setStockCode(selectedOption ? selectedOption.value : '')}
              options={stockCodes.map(code => ({ value: code, label: code }))}
              placeholder="全部"
              isClearable
              className="react-select-container"
              classNamePrefix="react-select"
              styles={{
                control: (base: any) => ({
                  ...base,
                  border: '1px solid #d9d9d9',
                  borderRadius: '4px',
                  boxShadow: 'none',
                  '&:hover': {
                    borderColor: '#1890ff',
                  },
                })
              }}
            />
          </div>
          <div className="filter-group">
            <label htmlFor="nameFilter">名称过滤</label>
            <input
              type="text"
              id="nameFilter"
              value={nameFilter}
              onChange={(e) => setNameFilter(e.target.value)}
              placeholder="正则表达式，如 SH[0-9]{6}"
              className="form-control"
            />
          </div>
          <div className="filter-group">
            <label htmlFor="expressionFilter">表达式过滤</label>
            <input
              type="text"
              id="expressionFilter"
              value={expressionFilter}
              onChange={(e) => setExpressionFilter(e.target.value)}
              placeholder="如 $close > 100"
              className="form-control"
            />
          </div>
          <div className="filter-group">
            <label htmlFor="startDate">开始日期</label>
            <input
              type="date"
              id="startDate"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="form-control"
            />
          </div>
          <div className="filter-group">
            <label htmlFor="endDate">结束日期</label>
            <input
              type="date"
              id="endDate"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="form-control"
            />
          </div>
          <div className="filter-actions">
            <button 
              className="btn btn-primary" 
              onClick={handleFilter}
              disabled={filtering}
            >
              {filtering ? '筛选中...' : '筛选'}
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={handleReset}
            >
              重置
            </button>
          </div>
        </div>
      </div>

      {/* 特征选择面板 */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="table-header">
          <h2>特征选择</h2>
          <div className="table-controls">
            <button 
              className="btn btn-sm btn-primary"
              onClick={() => setShowFeatureSelector(!showFeatureSelector)}
            >
              {showFeatureSelector ? '关闭特征选择' : '打开特征选择'}
            </button>
          </div>
        </div>
        
        {showFeatureSelector && (
          <div className="feature-selector">
            <div className="feature-list">
              <h3>可用特征</h3>
              <div className="feature-items">
                {availableFeatures.map(feature => (
                  <div key={feature.name} className="feature-item">
                    <label className="feature-checkbox">
                      <input
                        type="checkbox"
                        checked={selectedFeatures.includes(feature.name)}
                        onChange={() => toggleFeature(feature.name)}
                      />
                      <span className="feature-name">{feature.name}</span>
                      <span className="feature-description">{feature.description}</span>
                    </label>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="custom-feature">
              <h3>添加自定义特征</h3>
              <div className="custom-feature-form">
                <div className="form-group">
                  <label htmlFor="customFeatureName">特征名称</label>
                  <input
                    type="text"
                    id="customFeatureName"
                    value={customFeatureName}
                    onChange={(e) => setCustomFeatureName(e.target.value)}
                    placeholder="如：ma5"
                    className="form-control"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="customFeature">特征表达式</label>
                  <input
                    type="text"
                    id="customFeature"
                    value={customFeature}
                    onChange={(e) => setCustomFeature(e.target.value)}
                    placeholder="如：Mean($close, 5)"
                    className="form-control"
                  />
                </div>
                <button 
                  className="btn btn-primary"
                  onClick={addCustomFeature}
                  disabled={!customFeature || !customFeatureName}
                >
                  添加特征
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* 图表面板 */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="table-header">
          <h2>数据可视化</h2>
          <div className="table-controls">
            <button 
              className="btn btn-sm btn-primary"
              onClick={() => setShowChart(!showChart)}
            >
              {showChart ? '关闭图表' : '打开图表'}
            </button>
          </div>
        </div>
        
        {showChart && (
          <div className="chart-container">
            <div className="chart-controls">
              <div className="chart-control-group">
                <label htmlFor="chartStock">选择股票</label>
                <Select
              id="chartStock"
              value={chartStock ? { value: chartStock, label: chartStock } : null}
              onChange={(selectedOption: { value: string; label: string } | null) => setChartStock(selectedOption ? selectedOption.value : '')}
              options={stockCodes.map(code => ({ value: code, label: code }))}
              placeholder="请选择股票"
              isClearable
              className="react-select-container"
              classNamePrefix="react-select"
              styles={{
                control: (base: any) => ({
                  ...base,
                  border: '1px solid #d9d9d9',
                  borderRadius: '4px',
                  boxShadow: 'none',
                  '&:hover': {
                    borderColor: '#1890ff',
                  },
                })
              }}
            />
              </div>
              <div className="chart-control-group">
                <label htmlFor="chartType">图表类型</label>
                <select
                  id="chartType"
                  value={chartType}
                  onChange={(e) => setChartType(e.target.value as 'line' | 'candlestick' | 'bar' | 'multi' | 'technical' | 'heatmap' | 'distribution')}
                  className="form-control"
                >
                  <option value="line">折线图</option>
                  <option value="candlestick">K线图</option>
                  <option value="bar">柱状图</option>
                  <option value="multi">多股票对比</option>
                  <option value="technical">技术指标</option>
                  <option value="heatmap">相关性热力图</option>
                  <option value="distribution">分布分析</option>
                </select>
              </div>
              <div className="chart-control-group">
                <label htmlFor="chartFeature">选择特征</label>
                <select
                  id="chartFeature"
                  value={chartFeature}
                  onChange={(e) => setChartFeature(e.target.value)}
                  className="form-control"
                >
                  {selectedFeatures.map(feature => (
                    <option key={feature} value={feature}>
                      {feature}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            
            <div className="chart-wrapper">
            {chartType === 'multi' ? (
              chartStocks.length > 0 ? (
                <>
                  <div className="multi-stock-selection">
                    <h4>选择要比较的股票（最多5只）</h4>
                    <div className="stock-selection-list">
                      {stockCodes.slice(0, 20).map(code => (
                        <label key={code} className="stock-selection-item">
                          <input
                            type="checkbox"
                            checked={chartStocks.includes(code)}
                            onChange={() => toggleStockSelection(code)}
                          />
                          {code}
                        </label>
                      ))}
                    </div>
                    {chartStocks.length > 0 && (
                      <div className="selected-stocks">
                        <h5>已选择：{chartStocks.join(', ')}</h5>
                      </div>
                    )}
                  </div>
                  <MultiStockChart
                    data={chartStocks.reduce((acc, code) => {
                      acc[code] = stockData.filter(item => item.stock_code === code);
                      return acc;
                    }, {} as Record<string, StockData[]>)}
                    feature={chartFeature}
                    height="500px"
                  />
                </>
              ) : (
                <div className="chart-placeholder">
                  请选择至少一只股票进行比较
                </div>
              )
            ) : chartType === 'technical' ? (
              chartStock ? (
                <>
                  <div className="technical-indicators-selection">
                    <h4>选择技术指标</h4>
                    <div className="indicators-selection-list">
                      {['MA5', 'MA10', 'MA20', 'RSI', 'MACD'].map(indicator => (
                        <label key={indicator} className="indicator-selection-item">
                          <input
                            type="checkbox"
                            checked={selectedIndicators.includes(indicator)}
                            onChange={() => toggleIndicator(indicator)}
                          />
                          {indicator}
                        </label>
                      ))}
                    </div>
                    {selectedIndicators.length === 0 && (
                      <div className="no-indicators-selected">
                        请选择至少一个技术指标
                      </div>
                    )}
                  </div>
                  <TechnicalIndicatorsChart
                    data={stockData.filter(item => item.stock_code === chartStock)}
                    stockCode={chartStock}
                    indicators={selectedIndicators}
                    height="600px"
                  />
                </>
              ) : (
                <div className="chart-placeholder">
                  请选择股票以显示技术指标
                </div>
              )
            ) : chartType === 'heatmap' ? (
              <HeatmapChart
                data={stockData}
                features={selectedFeatures}
                height="600px"
              />
            ) : chartType === 'distribution' ? (
              chartFeature ? (
                <>
                  <div className="distribution-chart-controls">
                    <h4>选择图表类型</h4>
                    <div className="distribution-type-selector">
                      <label>
                        <input
                          type="radio"
                          value="histogram"
                          checked={distributionChartType === 'histogram'}
                          onChange={() => setDistributionChartType('histogram')}
                        />
                        直方图
                      </label>
                      <label>
                        <input
                          type="radio"
                          value="boxplot"
                          checked={distributionChartType === 'boxplot'}
                          onChange={() => setDistributionChartType('boxplot')}
                        />
                        箱线图
                      </label>
                    </div>
                  </div>
                  <DistributionChart
                    data={stockData}
                    feature={chartFeature}
                    chartType={distributionChartType}
                    height="400px"
                  />
                </>
              ) : (
                <div className="chart-placeholder">
                  请选择特征以显示分布图表
                </div>
              )
            ) : chartStock && chartFeature ? (
              chartType === 'line' ? (
                <LineChart
                  data={stockData.filter(item => item.stock_code === chartStock)}
                  stockCode={chartStock}
                  feature={chartFeature}
                  height="400px"
                />
              ) : chartType === 'candlestick' ? (
                <CandlestickChart
                  data={stockData.filter(item => item.stock_code === chartStock)}
                  stockCode={chartStock}
                  height="400px"
                />
              ) : (
                <BarChart
                  data={stockData.filter(item => item.stock_code === chartStock)}
                  stockCode={chartStock}
                  feature={chartFeature}
                  height="400px"
                />
              )
            ) : (
              <div className="chart-placeholder">
                请选择股票和特征以显示图表
              </div>
            )}
          </div>
          </div>
        )}
      </div>
      
      <div className="card">
        <div className="table-header">
          <h2>股票数据</h2>
          <div className="table-controls">
            <div className="per-page-selector">
              <label htmlFor="perPage">每页显示：</label>
              <select
                id="perPage"
                value={perPage}
                onChange={handlePerPageChange}
                className="form-control"
              >
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={200}>200</option>
                <option value={500}>500</option>
              </select>
            </div>
            <button 
              className="btn btn-sm btn-primary"
              onClick={exportToCSV}
              disabled={stockData.length === 0}
            >
              导出CSV
            </button>
          </div>
        </div>
        
        {loading ? (
          <div className="loading">加载中...</div>
        ) : (
          <>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>股票代码</th>
                    <th>日期</th>
                    {selectedFeatures.map(feature => (
                      <th key={feature}>{feature}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {stockData.map((data) => (
                    <tr key={data.id} onClick={() => setChartStock(data.stock_code)} className="data-row">
                      <td style={{ fontWeight: chartStock === data.stock_code ? 'bold' : 'normal', color: chartStock === data.stock_code ? '#1890ff' : 'inherit' }}>{data.stock_code}</td>
                      <td>{data.date}</td>
                      {selectedFeatures.map(feature => (
                        <td key={feature}>
                          {typeof data[feature] === 'number' ? data[feature].toFixed(2) : data[feature]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {total > 0 && (
              <div className="pagination">
                <div className="pagination-info">
                  显示 {((page - 1) * perPage) + 1} 到 {Math.min(page * perPage, total)} 条，共 {total} 条
                </div>
                <div className="pagination-controls">
                  <button 
                    className="btn btn-sm btn-secondary" 
                    onClick={() => handlePageChange(1)}
                    disabled={page === 1}
                  >
                    首页
                  </button>
                  <button 
                    className="btn btn-sm btn-secondary" 
                    onClick={() => handlePageChange(page - 1)}
                    disabled={page === 1}
                  >
                    上一页
                  </button>
                  <span className="page-info">
                    第 {page} 页 / 共 {totalPages} 页
                  </span>
                  <button 
                    className="btn btn-sm btn-secondary" 
                    onClick={() => handlePageChange(page + 1)}
                    disabled={page === totalPages}
                  >
                    下一页
                  </button>
                  <button 
                    className="btn btn-sm btn-secondary" 
                    onClick={() => handlePageChange(totalPages)}
                    disabled={page === totalPages}
                  >
                    末页
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default DataManagement
