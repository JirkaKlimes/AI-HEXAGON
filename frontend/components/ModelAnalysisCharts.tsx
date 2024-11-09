'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  TooltipProps,
} from 'recharts';
import { ModelData, Suite } from '@/app/utils/types';

interface ProcessedModelData {
  name: string;
  avgMetric: number;
  sizeMB: number;
  flopsGLOPs: number;
  sizeDoublingRate: number;
  flopsDoublingRate: number;
}

interface ModelAnalysisChartsProps {
  models: ModelData[];
  suite: Suite;
}

const ModelAnalysisCharts: React.FC<ModelAnalysisChartsProps> = ({
  models,
  suite,
}) => {
  // Process models data to select the best variation per model
  const processedModels: ProcessedModelData[] = models.map((model) => {
    const variations = Object.entries(model.variations);

    let bestVariationName = '';
    let bestVariation: any = null;
    let bestAvgMetric = -Infinity;

    variations.forEach(([variationName, variation]) => {
      const metricValues = Object.values(variation.metrics) as number[];
      const avgMetric =
        (metricValues.reduce((a, b) => a + b, 0) / metricValues.length) * 100;
      if (avgMetric > bestAvgMetric) {
        bestAvgMetric = avgMetric;
        bestVariationName = variationName;
        bestVariation = variation;
      }
    });

    return {
      name: model.title,
      avgMetric: bestAvgMetric,
      sizeMB: bestVariation.model_stats.size / (1024 * 1024),
      flopsGLOPs: bestVariation.model_stats.flops / 1e9,
      sizeDoublingRate: bestVariation.model_stats.size_doubling_rate,
      flopsDoublingRate: bestVariation.model_stats.flops_doubling_rate,
    };
  });

  const chartConfig = {
    height: 300,
    gridColor: '#4B5563', // Lighter grid color for better contrast
    tooltipStyle: {
      backgroundColor: '#1f2937',
      border: '1px solid #374151',
      color: '#e5e7eb',
    },
    axisStyle: {
      stroke: '#D1D5DB', // Lighter axis color
      fontSize: 12,
    },
    dotSize: 80, // Increase dot size
    dotStroke: '#FFFFFF', // White border around dots
  };

  const CustomTooltip: React.FC<TooltipProps<number, string>> = ({
    active,
    payload,
    label,
  }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload as ProcessedModelData;
      return (
        <div
          style={{
            backgroundColor: chartConfig.tooltipStyle.backgroundColor,
            border: chartConfig.tooltipStyle.border,
            color: chartConfig.tooltipStyle.color,
            padding: '10px',
          }}
        >
          <p style={{ margin: 0, fontWeight: 'bold' }}>{data.name}</p>
          {payload.map((entry, index) => (
            <p key={`item-${index}`} style={{ margin: 0 }}>
              {`${entry.name}: ${entry.value!.toFixed(2)}${entry.unit || ''}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const renderChart = (
    xDataKey: keyof ProcessedModelData,
    xLabel: string,
    xUnit: string,
    color: string
  ) => (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke={chartConfig.gridColor} />
          <XAxis
            type="number"
            dataKey={xDataKey}
            name={xLabel}
            unit={xUnit}
            tickFormatter={(value) => `${value.toFixed(2)}`}
            stroke={chartConfig.axisStyle.stroke}
            tick={{
              fill: chartConfig.axisStyle.stroke,
              fontSize: chartConfig.axisStyle.fontSize,
            }}
            label={{
              value: xLabel + (xUnit ? ` (${xUnit})` : ''),
              position: 'insideBottom',
              offset: -5,
              fill: chartConfig.axisStyle.stroke,
            }}
          />
          <YAxis
            type="number"
            dataKey="avgMetric"
            name="Average Metric"
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
            stroke={chartConfig.axisStyle.stroke}
            tick={{
              fill: chartConfig.axisStyle.stroke,
              fontSize: chartConfig.axisStyle.fontSize,
            }}
            label={{
              angle: -90,
              position: 'insideLeft',
              fill: chartConfig.axisStyle.stroke,
            }}
          />
          <RechartsTooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={<CustomTooltip />}
          />
          <Scatter
            name="Models"
            data={processedModels}
            fill={color}
            stroke={chartConfig.dotStroke}
            strokeWidth={1}
            shape="circle"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="space-y-6">
      <Card className="w-full bg-gradient-to-b from-gray-800 to-gray-900">
        <CardHeader>
          <CardTitle className="text-xl text-white">
            Model Performance at Sequence Length {suite.sequence_length}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-300">
                Size vs Performance
              </h3>
              {renderChart('sizeMB', 'Model Size', 'MB', '#2563eb')}
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-300">
                Compute vs Performance
              </h3>
              {renderChart(
                'flopsGLOPs',
                'Compute Requirements',
                'GLOPs',
                '#10b981'
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="w-full bg-gradient-to-b from-gray-800 to-gray-900">
        <CardHeader>
          <CardTitle className="text-xl text-white">
            Model Scaling Analysis
          </CardTitle>
          <p className="text-sm text-gray-400">
            Tested on sequence lengths: {suite.sequence_lengths.join(' • ')}
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-300">
                Size Scaling Efficiency
              </h3>
              {renderChart(
                'sizeDoublingRate',
                'Size Doubling Rate',
                '',
                '#2563eb'
              )}
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-300">
                Compute Scaling Efficiency
              </h3>
              {renderChart(
                'flopsDoublingRate',
                'Flops Doubling Rate',
                '',
                '#10b981'
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelAnalysisCharts;
