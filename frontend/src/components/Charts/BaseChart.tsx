import React, { useRef } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface BaseChartProps {
  data: any[];
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
  [key: string]: any;
}

const BaseChart: React.FC<BaseChartProps> = ({
  data,
  title,
  width = '100%',
  height = '400px',
  onChartReady,
  ...props
}) => {
  const chartRef = useRef<EChartsInstance | null>(null);

  const handleChartReady = (chartInstance: EChartsInstance) => {
    chartRef.current = chartInstance;
    if (onChartReady) {
      onChartReady(chartInstance);
    }
  };

  // Default chart option
  const getDefaultOption = (): EChartsOption => {
    return {
      title: {
        text: title,
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
      ...props
    };
  };

  return (
    <ReactECharts
      option={getDefaultOption()}
      style={{ height, width }}
      onChartReady={handleChartReady}
    />
  );
};

export default BaseChart;
