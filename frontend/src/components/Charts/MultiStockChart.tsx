import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface StockData {
  [key: string]: string | number;
}

interface MultiStockChartProps {
  data: Record<string, StockData[]>;
  feature: string;
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
}

const MultiStockChart: React.FC<MultiStockChartProps> = ({
  data,
  feature,
  title,
  width,
  height,
  onChartReady
}) => {
  // Define color palette for different stocks
  const colors = [
    '#1890ff', '#2fc25b', '#facc14', '#223273', '#8543e0',
    '#13c2c2', '#eb2f96', '#fa8c16', '#a0d911', '#52c41a'
  ];

  // Get all unique dates from all stocks
  const allDates = new Set<string>();
  Object.values(data).forEach(stockData => {
    stockData.forEach(item => {
      allDates.add(item.date as string);
    });
  });

  // Sort dates
  const sortedDates = Array.from(allDates).sort((a, b) => {
    const dateA = new Date(a).getTime();
    const dateB = new Date(b).getTime();
    return dateA - dateB;
  });

  // Prepare series data for each stock
  const series = Object.entries(data).map(([stockCode, stockData], index) => {
    // Create a map from date to value for quick lookup
    const dateValueMap = new Map<string, number>();
    stockData.forEach(item => {
      dateValueMap.set(item.date as string, item[feature] as number);
    });

    // Fill in missing values with null (will be skipped in the chart)
    const values = sortedDates.map(date => dateValueMap.get(date) || null);

    return {
      name: stockCode,
      type: 'line',
      data: values,
      smooth: true,
      lineStyle: {
        width: 2,
        color: colors[index % colors.length]
      },
      itemStyle: {
        color: colors[index % colors.length],
        borderWidth: 2
      },
      emphasis: {
        focus: 'series',
        itemStyle: {
          borderWidth: 4
        }
      },
      animationDuration: 1000
    };
  });

  const getChartOption = (): EChartsOption => {
    return {
      title: {
        text: title || `多股票 ${feature} 对比图`,
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
        data: Object.keys(data),
        top: 'bottom',
        type: 'scroll',
        textStyle: {
          fontSize: 10
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%', // Increase bottom margin for legend
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: sortedDates,
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
      series: series
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

export default MultiStockChart;
