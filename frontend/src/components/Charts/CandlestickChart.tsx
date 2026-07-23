import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface StockData {
  [key: string]: string | number;
}

interface CandlestickChartProps {
  data: StockData[];
  stockCode: string;
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  stockCode,
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

  const getChartOption = (): EChartsOption => {
    return {
      title: {
        text: title || `${stockCode} K线图`,
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
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
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
        }
      },
      yAxis: {
        type: 'value',
        axisLine: {
          lineStyle: {
            color: '#ccc'
          }
        },
        splitLine: {
          lineStyle: {
            color: '#f0f0f0'
          }
        }
      },
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
          }
        }
      ]
    };
  };

  return (
    <ReactECharts
      option={getChartOption()}
      style={{ height: height || '400px', width: width || '100%' }}
      onChartReady={onChartReady}
    />
  );
};

export default CandlestickChart;
