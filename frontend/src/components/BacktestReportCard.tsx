import React from 'react'
import type { BacktestAnalysis as BacktestAnalysisType } from '../services/experiments'

interface BacktestReportCardProps {
  data: BacktestAnalysisType
}

const BacktestReportCard: React.FC<BacktestReportCardProps> = ({ data }) => {
  return (
    <div className="backtest-report-container">
      <h2>Backtest Return Analysis</h2>
      
      <div className="report-card">
        <h3>Report Summary</h3>
        <div className="metrics-grid">
          <div className="metric-item">
            <span className="metric-label">Total Return</span>
            <span className="metric-value">{(data.report.total_return * 100).toFixed(2)}%</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Annual Return</span>
            <span className="metric-value">{(data.report.annual_return * 100).toFixed(2)}%</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Sharpe Ratio</span>
            <span className="metric-value">{data.report.sharpe_ratio.toFixed(2)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Max Drawdown</span>
            <span className="metric-value">{(data.report.max_drawdown * 100).toFixed(2)}%</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Win Rate</span>
            <span className="metric-value">{(data.report.win_rate * 100).toFixed(2)}%</span>
          </div>
        </div>
      </div>
      
      <div className="explanation-card">
        <h3>Explanation of Results</h3>
        <p>{data.explanation}</p>
      </div>
    </div>
  )
}

export default BacktestReportCard