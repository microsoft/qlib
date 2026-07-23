import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, EChartsInstance } from 'echarts-for-react';

interface StockData {
  [key: string]: string | number;
}

interface HeatmapChartProps {
  data: StockData[];
  features: string[];
  title?: string;
  width?: string | number;
  height?: string | number;
  onChartReady?: (chartInstance: EChartsInstance) => void;
}

const HeatmapChart: React.FC<HeatmapChartProps> = ({
  data,
  features,
  title,
  width,
  height,
  onChartReady
}) => {
  // Calculate correlation matrix
  const calculateCorrelationMatrix = () => {
    const matrix: number[][] = [];
    const featureCount = features.length;

    // Initialize matrix with zeros
    for (let i = 0; i < featureCount; i++) {
      matrix[i] = new Array(featureCount).fill(0);
    }

    // Calculate correlation between each pair of features
    for (let i = 0; i < featureCount; i++) {
      for (let j = 0; j < featureCount; j++) {
        if (i === j) {
          // Correlation with itself is 1
          matrix[i][j] = 1;
        } else if (i < j) {
          // Calculate correlation only once and mirror it
          const correlation = calculateCorrelation(features[i], features[j]);
          matrix[i][j] = correlation;
          matrix[j][i] = correlation;
        }
      }
    }

    return matrix;
  };

  // Helper function to calculate correlation between two features
  const calculateCorrelation = (feature1: string, feature2: string): number => {
    const values1 = data.map(item => item[feature1] as number).filter(val => !isNaN(val));
    const values2 = data.map(item => item[feature2] as number).filter(val => !isNaN(val));

    const n = Math.min(values1.length, values2.length);
    if (n < 2) {
      return 0;
    }

    // Calculate means
    const mean1 = values1.slice(0, n).reduce((sum, val) => sum + val, 0) / n;
    const mean2 = values2.slice(0, n).reduce((sum, val) => sum + val, 0) / n;

    // Calculate covariance and variances
    let covariance = 0;
    let variance1 = 0;
    let variance2 = 0;

    for (let i = 0; i < n; i++) {
      const diff1 = values1[i] - mean1;
      const diff2 = values2[i] - mean2;

      covariance += diff1 * diff2;
      variance1 += diff1 * diff1;
      variance2 += diff2 * diff2;
    }

    covariance /= n - 1;
    variance1 /= n - 1;
    variance2 /= n - 1;

    // Calculate correlation coefficient
    const stdDev1 = Math.sqrt(variance1);
    const stdDev2 = Math.sqrt(variance2);

    if (stdDev1 === 0 || stdDev2 === 0) {
      return 0;
    }

    return covariance / (stdDev1 * stdDev2);
  };

  const correlationMatrix = calculateCorrelationMatrix();

  // Prepare data for heatmap
  const heatmapData: [number, number, number][] = [];
  for (let i = 0; i < features.length; i++) {
    for (let j = 0; j < features.length; j++) {
      heatmapData.push([j, i, correlationMatrix[i][j]]);
    }
  }

  const getChartOption = (): EChartsOption => {
    return {
      title: {
        text: title || '特征相关性热力图',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        position: 'top',
        formatter: function(params: any) {
          const feature1 = features[params.value[1]];
          const feature2 = features[params.value[0]];
          const correlation = params.value[2].toFixed(4);
          return `${feature1} vs ${feature2}<br/>相关性: ${correlation}`;
        }
      },
      grid: {
        height: '50%',
        top: '15%'
      },
      xAxis: {
        type: 'category',
        data: features,
        axisLabel: {
          rotate: 45,
          fontSize: 10
        },
        splitArea: {
          show: true
        }
      },
      yAxis: {
        type: 'category',
        data: features,
        axisLabel: {
          fontSize: 10
        },
        splitArea: {
          show: true
        }
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%',
        inRange: {
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        },
        textStyle: {
          fontSize: 10
        }
      },
      series: [
        {
          name: '相关性',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: function(params: any) {
              return params.value[2].toFixed(2);
            },
            fontSize: 8
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    };
  };

  return (
    <ReactECharts
      option={getChartOption()}
      style={{ height: height || '500px', width: width || '100%' }}
      onChartReady={onChartReady}
    />
  );
};

export default HeatmapChart;
