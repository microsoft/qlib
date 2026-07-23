import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface StockData {
  [key: string]: string | number;
}

interface TechnicalIndicatorsChartProps {
  data: StockData[];
  stockCode: string;
  indicators: string[];
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
}

const TechnicalIndicatorsChart: React.FC<TechnicalIndicatorsChartProps> = ({
  data,
  stockCode,
  indicators,
  title,
  width,
  height,
  onChartReady
}) => {
  // Sort data by date
  const sortedData = [...data].sort((a, b) => {
    const dateA = new Date(a.date as string).getTime();
    const dateB = new Date(b.date as string).getTime();
    return dateA - dateB;
  });

  const dates = sortedData.map(item => item.date as string);
  const candlestickData = sortedData.map(item => [
    item.open as number,
    item.close as number,
    item.low as number,
    item.high as number
  ]);

  // Calculate technical indicators
  const calculateIndicators = () => {
    const results: any = {};

    // Calculate moving averages
    if (indicators.includes('MA5') || indicators.includes('MA10') || indicators.includes('MA20')) {
      const closePrices = sortedData.map(item => item.close as number);
      
      // MA5
      if (indicators.includes('MA5')) {
        results.MA5 = calculateMovingAverage(closePrices, 5);
      }
      
      // MA10
      if (indicators.includes('MA10')) {
        results.MA10 = calculateMovingAverage(closePrices, 10);
      }
      
      // MA20
      if (indicators.includes('MA20')) {
        results.MA20 = calculateMovingAverage(closePrices, 20);
      }
    }

    // Calculate RSI (14 periods)
    if (indicators.includes('RSI')) {
      const closePrices = sortedData.map(item => item.close as number);
      results.RSI = calculateRSI(closePrices, 14);
    }

    // Calculate MACD
    if (indicators.includes('MACD')) {
      const closePrices = sortedData.map(item => item.close as number);
      results.MACD = calculateMACD(closePrices);
    }

    return results;
  };

  // Helper function to calculate moving average
  const calculateMovingAverage = (prices: number[], period: number): (number | null)[] => {
    const result: (number | null)[] = [];
    for (let i = 0; i < prices.length; i++) {
      if (i < period - 1) {
        result.push(null);
      } else {
        const sum = prices.slice(i - period + 1, i + 1).reduce((acc, val) => acc + val, 0);
        result.push(sum / period);
      }
    }
    return result;
  };

  // Helper function to calculate RSI
  const calculateRSI = (prices: number[], period: number): (number | null)[] => {
    const result: (number | null)[] = [];
    const gains: number[] = [];
    const losses: number[] = [];

    // Calculate initial gains and losses
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }

    // Calculate initial average gain and loss
    let avgGain = gains.slice(0, period).reduce((acc, val) => acc + val, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((acc, val) => acc + val, 0) / period;

    // Calculate initial RSI
    result.push(null); // For the first price point
    for (let i = 0; i < period; i++) {
      result.push(null);
    }
    
    if (avgLoss === 0) {
      result.push(100);
    } else {
      const rs = avgGain / avgLoss;
      result.push(100 - (100 / (1 + rs)));
    }

    // Calculate RSI for remaining periods
    for (let i = period; i < gains.length; i++) {
      avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
      avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
      
      if (avgLoss === 0) {
        result.push(100);
      } else {
        const rs = avgGain / avgLoss;
        result.push(100 - (100 / (1 + rs)));
      }
    }

    return result;
  };

  // Helper function to calculate MACD
  const calculateMACD = (prices: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): any => {
    const macdLine: (number | null)[] = [];
    const signalLine: (number | null)[] = [];
    const histogram: (number | null)[] = [];

    // Calculate fast EMA
    const fastEMA = calculateEMA(prices, fastPeriod);
    
    // Calculate slow EMA
    const slowEMA = calculateEMA(prices, slowPeriod);
    
    // Calculate MACD line
    for (let i = 0; i < prices.length; i++) {
      if (i < slowPeriod - 1) {
        macdLine.push(null);
      } else {
        macdLine.push(fastEMA[i] - slowEMA[i]);
      }
    }
    
    // Calculate signal line (EMA of MACD line)
    for (let i = 0; i < macdLine.length; i++) {
      if (i < slowPeriod + signalPeriod - 2) {
        signalLine.push(null);
      } else {
        const signalEMA = calculateEMA(macdLine.slice(i - signalPeriod + 1, i + 1).filter(val => val !== null) as number[], signalPeriod);
        signalLine.push(signalEMA[signalEMA.length - 1]);
      }
    }
    
    // Calculate histogram
    for (let i = 0; i < macdLine.length; i++) {
      if (macdLine[i] === null || signalLine[i] === null) {
        histogram.push(null);
      } else {
        histogram.push((macdLine[i] as number) - (signalLine[i] as number));
      }
    }
    
    return { macdLine, signalLine, histogram };
  };

  // Helper function to calculate EMA
  const calculateEMA = (prices: number[], period: number): number[] => {
    const result: number[] = [];
    const multiplier = 2 / (period + 1);
    
    // Calculate initial SMA
    let sma = prices.slice(0, period).reduce((acc, val) => acc + val, 0) / period;
    result.push(sma);
    
    // Calculate EMA for remaining periods
    for (let i = period; i < prices.length; i++) {
      const ema = (prices[i] - result[i - period]) * multiplier + result[i - period];
      result.push(ema);
    }
    
    // Fill in the beginning with nulls
    const nulls = Array(period - 1).fill(null);
    return [...nulls, ...result];
  };

  const indicatorsData = calculateIndicators();

  const getChartOption = (): EChartsOption => {
    const option: EChartsOption = {
      title: {
        text: title || `${stockCode} 技术指标图`,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          label: {
            backgroundColor: '#6a7985'
          }
        }
      },
      legend: {
        data: ['K线', ...indicators],
        top: 'bottom',
        type: 'scroll',
        textStyle: {
          fontSize: 10
        }
      },
      grid: [
        {
          left: '3%',
          right: '4%',
          height: '50%',
          containLabel: true
        },
        {
          left: '3%',
          right: '4%',
          top: '60%',
          height: '30%',
          containLabel: true
        }
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          axisLabel: {
            rotate: 45,
            fontSize: 10
          },
          axisLine: {
            lineStyle: {
              color: '#ccc'
            }
          },
          gridIndex: 0
        },
        {
          type: 'category',
          data: dates,
          axisLabel: {
            show: false
          },
          axisLine: {
            show: false
          },
          axisTick: {
            show: false
          },
          gridIndex: 1
        }
      ],
      yAxis: [
        {
          type: 'value',
          scale: true,
          axisLine: {
            lineStyle: {
              color: '#ccc'
            }
          },
          splitLine: {
            lineStyle: {
              color: '#f0f0f0'
            }
          },
          gridIndex: 0
        },
        {
          type: 'value',
          scale: true,
          axisLine: {
            lineStyle: {
              color: '#ccc'
            }
          },
          splitLine: {
            lineStyle: {
              color: '#f0f0f0'
            }
          },
          gridIndex: 1
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          data: candlestickData,
          itemStyle: {
            color: '#ec0000',
            color0: '#00da3c',
            borderColor: '#ec0000',
            borderColor0: '#00da3c'
          },
          emphasis: {
            itemStyle: {
              color: '#ff7f50',
              color0: '#87ceeb',
              borderColor: '#ff7f50',
              borderColor0: '#87ceeb'
            }
          },
          xAxisIndex: 0,
          yAxisIndex: 0
        }
      ]
    };

    // Add moving averages to the chart
    if (indicatorsData.MA5) {
      option.series?.push({
        name: 'MA5',
        type: 'line',
        data: indicatorsData.MA5,
        smooth: true,
        lineStyle: {
          width: 1,
          color: '#FFA500'
        },
        itemStyle: {
          color: '#FFA500'
        },
        emphasis: {
          focus: 'series'
        },
        xAxisIndex: 0,
        yAxisIndex: 0
      });
    }

    if (indicatorsData.MA10) {
      option.series?.push({
        name: 'MA10',
        type: 'line',
        data: indicatorsData.MA10,
        smooth: true,
        lineStyle: {
          width: 1,
          color: '#FF6347'
        },
        itemStyle: {
          color: '#FF6347'
        },
        emphasis: {
          focus: 'series'
        },
        xAxisIndex: 0,
        yAxisIndex: 0
      });
    }

    if (indicatorsData.MA20) {
      option.series?.push({
        name: 'MA20',
        type: 'line',
        data: indicatorsData.MA20,
        smooth: true,
        lineStyle: {
          width: 1,
          color: '#4682B4'
        },
        itemStyle: {
          color: '#4682B4'
        },
        emphasis: {
          focus: 'series'
        },
        xAxisIndex: 0,
        yAxisIndex: 0
      });
    }

    // Add RSI to the chart
    if (indicatorsData.RSI) {
      option.series?.push({
        name: 'RSI',
        type: 'line',
        data: indicatorsData.RSI,
        smooth: true,
        lineStyle: {
          width: 2,
          color: '#9932CC'
        },
        itemStyle: {
          color: '#9932CC'
        },
        emphasis: {
          focus: 'series'
        },
        xAxisIndex: 1,
        yAxisIndex: 1
      });
    }

    // Add MACD to the chart
    if (indicatorsData.MACD) {
      option.series?.push({
        name: 'MACD',
        type: 'line',
        data: indicatorsData.MACD.macdLine,
        smooth: true,
        lineStyle: {
          width: 2,
          color: '#FF69B4'
        },
        itemStyle: {
          color: '#FF69B4'
        },
        emphasis: {
          focus: 'series'
        },
        xAxisIndex: 1,
        yAxisIndex: 1
      });

      option.series?.push({
        name: 'Signal',
        type: 'line',
        data: indicatorsData.MACD.signalLine,
        smooth: true,
        lineStyle: {
          width: 2,
          color: '#00BFFF'
        },
        itemStyle: {
          color: '#00BFFF'
        },
        emphasis: {
          focus: 'series'
        },
        xAxisIndex: 1,
        yAxisIndex: 1
      });

      option.series?.push({
        name: 'Histogram',
        type: 'bar',
        data: indicatorsData.MACD.histogram,
        itemStyle: {
          color: function(params: any) {
            return params.data >= 0 ? '#FF69B4' : '#00BFFF';
          }
        },
        emphasis: {
          focus: 'series'
        },
        xAxisIndex: 1,
        yAxisIndex: 1
      });
    }

    return option;
  };

  return (
    <ReactECharts
      option={getChartOption()}
      style={{ height: height || '600px', width: width || '100%' }}
      onChartReady={onChartReady}
    />
  );
};

export default TechnicalIndicatorsChart;
