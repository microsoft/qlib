import React from 'react'
import ReactECharts from 'echarts-for-react'
import type { PortfolioAnalysis as PortfolioAnalysisType } from '../services/experiments'

interface PortfolioAnalysisChartProps {
  data: PortfolioAnalysisType
}

const PortfolioAnalysisChart: React.FC<PortfolioAnalysisChartProps> = ({ data }) => {
  // Cumulative Return chart
  const getCumulativeReturnChartOption = () => {
    return {
      title: {
        text: 'Cumulative Return',
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
        data: data.cumulative_return.dates,
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
          data: data.cumulative_return.values.map(v => (v * 100).toFixed(2)),
          type: 'line',
          smooth: true,
          itemStyle: {
            color: '#646cff'
          }
        }
      ]
    }
  }

  // Group Returns chart
  const getGroupReturnsChartOption = () => {
    const series = Object.entries(data.group_returns.groups).map(([group, values], index) => {
      const colors = ['#646cff', '#74c0fc', '#91d5ff', '#b2e2ff', '#d1f0ff', '#7048e8', '#9775fa', '#b197fc', '#d0bfff', '#e5dbff']
      return {
        name: group,
        data: values.map(v => (v * 100).toFixed(2)),
        type: 'line',
        smooth: true,
        itemStyle: {
          color: colors[index % colors.length]
        }
      }
    })

    return {
      title: {
        text: 'Cumulative Return of Groups',
        left: 'center'
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
        data: Object.keys(data.group_returns.groups),
        orient: 'horizontal',
        bottom: 0
      },
      xAxis: {
        type: 'category',
        data: data.group_returns.dates,
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
      series
    }
  }

  // Long Short chart
  const getLongShortChartOption = () => {
    return {
      title: {
        text: 'Long-Short Strategy Return',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const date = params[0].axisValue
          const value = params[0].value
          return `${date}<br/>Return: ${(value * 100).toFixed(2)}%`
        }
      },
      xAxis: {
        type: 'category',
        data: data.long_short.dates,
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
          data: data.long_short.values.map(v => (v * 100).toFixed(2)),
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
    <div className="portfolio-analysis-container">
      <h2>Portfolio Analysis</h2>
      
      <div className="chart-grid">
        <div className="chart-item">
          <ReactECharts option={getCumulativeReturnChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
        
        <div className="chart-item">
          <ReactECharts option={getGroupReturnsChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
        
        <div className="chart-item">
          <ReactECharts option={getLongShortChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
      </div>
    </div>
  )
}

export default PortfolioAnalysisChart