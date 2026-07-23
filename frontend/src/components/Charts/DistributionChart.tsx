import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface StockData {
  [key: string]: string | number;
}

interface DistributionChartProps {
  data: StockData[];
  feature: string;
  chartType?: 'histogram' | 'boxplot';
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
}

const DistributionChart: React.FC<DistributionChartProps> = ({
  data,
  feature,
  chartType = 'histogram',
  title,
  width,
  height,
  onChartReady
}) => {
  // Extract values for the selected feature
  const values = data.map(item => item[feature] as number).filter(val => !isNaN(val));

  // Calculate histogram data
  const calculateHistogramData = () => {
    if (values.length === 0) {
      return [];
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const binCount = Math.ceil(Math.sqrt(values.length)); // Sturges' rule
    const binWidth = (max - min) / binCount;

    // Initialize bins
    const bins: number[] = new Array(binCount).fill(0);
    
    // Count values in each bin
    values.forEach(val => {
      const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
      bins[binIndex]++;
    });

    // Format data for ECharts
    return bins.map((count, index) => {
      const binStart = min + index * binWidth;
      const binEnd = binStart + binWidth;
      return {
        name: `${binStart.toFixed(2)} - ${binEnd.toFixed(2)}`,
        value: count
      };
    });
  };

  // Calculate box plot data
  const calculateBoxplotData = () => {
    if (values.length === 0) {
      return [];
    }

    // Sort values
    const sortedValues = [...values].sort((a, b) => a - b);
    const n = sortedValues.length;

    // Calculate quartiles
    const q1Index = Math.floor(n * 0.25);
    const q2Index = Math.floor(n * 0.5);
    const q3Index = Math.floor(n * 0.75);

    const q1 = sortedValues[q1Index];
    const q2 = sortedValues[q2Index]; // Median
    const q3 = sortedValues[q3Index];

    // Calculate interquartile range (IQR)
    const iqr = q3 - q1;

    // Calculate whiskers
    const lowerWhisker = Math.max(sortedValues[0], q1 - 1.5 * iqr);
    const upperWhisker = Math.min(sortedValues[n - 1], q3 + 1.5 * iqr);

    // Identify outliers
    const outliers = sortedValues.filter(val => val < lowerWhisker || val > upperWhisker);

    return {
      boxplot: [lowerWhisker, q1, q2, q3, upperWhisker],
      outliers
    };
  };

  const histogramData = calculateHistogramData();
  const boxplotData = calculateBoxplotData();

  const getChartOption = (): EChartsOption => {
    if (chartType === 'histogram') {
      return {
        title: {
          text: title || `${feature} 直方图`,
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
          },
          formatter: function(params: any) {
            return `${params[0].name}<br/>频数: ${params[0].value}`;
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
          data: histogramData.map(item => item.name),
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
          name: '频数',
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
        series: [
          {
            name: '频数',
            type: 'bar',
            data: histogramData.map(item => item.value),
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
    } else {
      // Boxplot
      return {
        title: {
          text: title || `${feature} 箱线图`,
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold'
          }
        },
        tooltip: {
          trigger: 'item',
          axisPointer: {
            type: 'shadow'
          },
          formatter: function(params: any) {
            if (params.seriesName === '异常值') {
              return `异常值: ${params.value}`;
            } else {
              const data = params.data;
              return `最小值: ${data[0]}<br/>Q1: ${data[1]}<br/>中位数: ${data[2]}<br/>Q3: ${data[3]}<br/>最大值: ${data[4]}`;
            }
          }
        },
        grid: {
          left: '10%',
          right: '10%',
          bottom: '15%'
        },
        xAxis: {
          type: 'category',
          data: [feature],
          axisLabel: {
            fontSize: 12
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
        series: [
          {
            name: '箱线图',
            type: 'boxplot',
            data: [(boxplotData as { boxplot: number[]; outliers: number[] }).boxplot],
            itemStyle: {
              color: '#1890ff'
            },
            emphasis: {
              itemStyle: {
                color: '#40a9ff'
              }
            }
          },
          {
            name: '异常值',
            type: 'scatter',
            data: (boxplotData as { boxplot: number[]; outliers: number[] }).outliers.map((val: number) => [feature, val]),
            itemStyle: {
              color: '#ff4d4f'
            },
            emphasis: {
              itemStyle: {
                color: '#ff7875'
              }
            }
          }
        ]
      };
    }
  };

  return (
    <ReactECharts
      option={getChartOption()}
      style={{ height: height || '400px', width: width || '100%' }}
      onChartReady={onChartReady}
    />
  );
};

export default DistributionChart;
