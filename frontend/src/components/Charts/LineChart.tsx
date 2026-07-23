import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface StockData {
  [key: string]: string | number;
}

interface LineChartProps {
  data: StockData[];
  stockCode: string;
  feature: string;
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
}

const LineChart: React.FC<LineChartProps> = ({
  data,
  stockCode,
  feature,
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
  const values = sortedData.map(item => item[feature] as number);

  const getChartOption = (): EChartsOption => {
    return {
      title: {
        text: title || `${stockCode} ${feature} 趋势图`,
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
          name: feature,
          type: 'line',
          data: values,
          smooth: true,
          lineStyle: {
            width: 2,
            color: '#1890ff'
          },
          itemStyle: {
            color: '#1890ff',
            borderWidth: 2
          },
          emphasis: {
            focus: 'series',
            itemStyle: {
              borderWidth: 4
            }
          },
          animationDuration: 1000
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

export default LineChart;
