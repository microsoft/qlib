import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface StockData {
  [key: string]: string | number;
}

interface BarChartProps {
  data: StockData[];
  stockCode: string;
  feature: string;
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
}

const BarChart: React.FC<BarChartProps> = ({
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
        text: title || `${stockCode} ${feature} 柱状图`,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
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
          type: 'bar',
          data: values,
          itemStyle: {
            color: '#1890ff',
            borderRadius: [4, 4, 0, 0]
          },
          emphasis: {
            itemStyle: {
              color: '#40a9ff'
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

export default BarChart;
