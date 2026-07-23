import React from 'react'
import ReactECharts from 'echarts-for-react'
import type { SignalAnalysis as SignalAnalysisType } from '../services/experiments'

interface SignalAnalysisChartProps {
  data: SignalAnalysisType
}

const SignalAnalysisChart: React.FC<SignalAnalysisChartProps> = ({ data }) => {
  // IC time series chart
  const getICChartOption = () => {
    return {
      title: {
        text: 'Information Coefficient (IC)',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const date = params[0].axisValue
          const value = params[0].value
          return `${date}<br/>IC: ${(value * 100).toFixed(2)}%`
        }
      },
      xAxis: {
        type: 'category',
        data: data.ic.dates,
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
          data: data.ic.values.map(v => (v * 100).toFixed(2)),
          type: 'line',
          smooth: true,
          itemStyle: {
            color: '#646cff'
          }
        }
      ]
    }
  }

  // Monthly IC chart
  const getMonthlyICChartOption = () => {
    return {
      title: {
        text: 'Monthly Information Coefficient',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const month = params[0].axisValue
          const value = params[0].value
          return `${month}<br/>IC: ${(value * 100).toFixed(2)}%`
        }
      },
      xAxis: {
        type: 'category',
        data: data.monthly_ic.months,
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
          data: data.monthly_ic.values.map(v => (v * 100).toFixed(2)),
          type: 'bar',
          itemStyle: {
            color: '#646cff'
          }
        }
      ]
    }
  }

  // Auto Correlation chart
  const getAutoCorrelationChartOption = () => {
    return {
      title: {
        text: 'Auto Correlation of Forecasting Signal',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const lag = params[0].axisValue
          const value = params[0].value
          return `Lag ${lag}<br/>Auto Correlation: ${(value * 100).toFixed(2)}%`
        }
      },
      xAxis: {
        type: 'category',
        data: data.auto_correlation.lags
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}%'
        }
      },
      series: [
        {
          data: data.auto_correlation.values.map(v => (v * 100).toFixed(2)),
          type: 'bar',
          itemStyle: {
            color: '#646cff'
          }
        }
      ]
    }
  }

  // Return Distribution chart
  const getReturnDistributionChartOption = () => {
    return {
      title: {
        text: 'Return Distribution',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const bin = params[0].axisValue
          const count = params[0].value
          return `Bin: ${bin}<br/>Count: ${count}`
        }
      },
      xAxis: {
        type: 'category',
        data: data.return_distribution.bins
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          data: data.return_distribution.counts,
          type: 'bar',
          itemStyle: {
            color: '#646cff'
          }
        }
      ]
    }
  }

  return (
    <div className="signal-analysis-container">
      <h2>Forecasting Signal Analysis</h2>
      
      <div className="chart-grid">
        <div className="chart-item">
          <ReactECharts option={getICChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
        
        <div className="chart-item">
          <ReactECharts option={getMonthlyICChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
        
        <div className="chart-item">
          <ReactECharts option={getAutoCorrelationChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
        
        <div className="chart-item">
          <ReactECharts option={getReturnDistributionChartOption()} style={{ height: '400px', width: '100%' }} />
        </div>
      </div>
    </div>
  )
}

export default SignalAnalysisChart