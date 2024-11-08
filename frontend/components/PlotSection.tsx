'use client';
import React, { FC, ReactNode } from 'react';
import { ModelData, Suite } from '@/app/utils/types';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts';

interface PlotSectionProps {
  results: ModelData[];
  suite: Suite;
}

interface DataPoint {
  name: string;
  avgMetric: number;
  size: number;
  sizeDoublingRate: number;
  flops: number;
  flopsDoublingRate: number;
}

interface Plot {
  xKey: keyof DataPoint;
  xLabel: string;
}

const ChartContainer: FC<{ children: ReactNode; title: string }> = ({
  children,
  title,
}) => (
  <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
    <h4 className="text-lg font-medium text-gray-200 mb-4">{title}</h4>
    {children}
  </div>
);
ChartContainer.displayName = 'ChartContainer';

const PlotSection: FC<PlotSectionProps> = ({ results, suite }) => {
  const data: DataPoint[] = results.map((model) => ({
    name: model.title,
    avgMetric:
      Object.values(model.metrics).reduce((a, b) => a + b, 0) /
      Object.keys(model.metrics).length,
    size: model.model_stats.size / 1e6,
    sizeDoublingRate: model.model_stats.size_doubling_rate,
    flops: model.model_stats.flops / 1e9,
    flopsDoublingRate: model.model_stats.flops_doubling_rate,
  }));

  const plots: Plot[] = [
    { xKey: 'size', xLabel: 'Model Size (MB)' },
    { xKey: 'flops', xLabel: 'FLOPs (GFLOPs)' },
    { xKey: 'sizeDoublingRate', xLabel: 'Size Doubling Rate' },
    { xKey: 'flopsDoublingRate', xLabel: 'FLOPs Doubling Rate' },
  ];

  const CustomTooltip = (xKey: keyof DataPoint, xLabel: string) => {
    const TooltipComponent: FC<TooltipProps<number, string>> = ({
      active,
      payload,
    }) => {
      if (active && payload && payload.length > 0 && payload[0].payload) {
        const dataPoint = payload[0].payload as DataPoint;
        const value = dataPoint[xKey];
        const formattedValue = xKey.includes('Rate')
          ? value.toLocaleString()
          : value.toLocaleString(undefined, {
              maximumFractionDigits: 2,
              minimumFractionDigits: 2,
            });

        return (
          <div className="bg-gray-800 shadow-xl p-3 rounded-lg border border-gray-700">
            <p className="font-medium text-white">{dataPoint.name}</p>
            <div className="mt-1 space-y-1 text-sm text-gray-300">
              <p>{`${xLabel}: ${formattedValue}`}</p>
              <p>{`Average Metric: ${dataPoint.avgMetric.toFixed(3)}`}</p>
            </div>
          </div>
        );
      }
      return null;
    };

    TooltipComponent.displayName = `CustomTooltip_${xKey}`;

    return TooltipComponent;
  };

  return (
    <section className="w-full max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="space-y-16">
        <div>
          <h3 className="text-2xl font-semibold text-white mb-6">
            Model Performance at Sequence Length {suite.sequence_length}
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {plots.slice(0, 2).map(({ xKey, xLabel }) => (
              <ChartContainer key={xKey} title={xLabel}>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey={xKey}
                      type="number"
                      name={xLabel}
                      tick={{ fill: '#9CA3AF' }}
                      tickLine={{ stroke: '#4B5563' }}
                      axisLine={{ stroke: '#4B5563' }}
                      label={{
                        value: xLabel,
                        position: 'bottom',
                        offset: 0,
                        style: { fill: '#9CA3AF', fontSize: 12 },
                      }}
                    />
                    <YAxis
                      dataKey="avgMetric"
                      type="number"
                      tick={{ fill: '#9CA3AF' }}
                      tickLine={{ stroke: '#4B5563' }}
                      axisLine={{ stroke: '#4B5563' }}
                      label={{
                        value: 'Average Metric',
                        angle: -90,
                        position: 'left',
                        offset: 0,
                        style: { fill: '#9CA3AF', fontSize: 12 },
                      }}
                    />
                    <Tooltip
                      content={CustomTooltip(xKey, xLabel)}
                      cursor={{ stroke: '#6B7280', strokeDasharray: '3 3' }}
                    />
                    <Scatter
                      data={data}
                      fill="#6366f1"
                      fillOpacity={0.8}
                      stroke="#818cf8"
                      strokeWidth={1}
                      r={6}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </ChartContainer>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-6">
            <h3 className="text-2xl font-semibold text-white">
              Model Scaling Analysis
            </h3>
            <p className="mt-2 text-gray-400">
              Tested on sequence lengths:{' '}
              {suite.sequence_lengths
                .map((len) => len.toLocaleString())
                .join(' • ')}
            </p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {plots.slice(2).map(({ xKey, xLabel }) => (
              <ChartContainer key={xKey} title={xLabel}>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey={xKey}
                      type="number"
                      name={xLabel}
                      tick={{ fill: '#9CA3AF' }}
                      tickLine={{ stroke: '#4B5563' }}
                      axisLine={{ stroke: '#4B5563' }}
                      label={{
                        value: xLabel,
                        position: 'bottom',
                        offset: 0,
                        style: { fill: '#9CA3AF', fontSize: 12 },
                      }}
                    />
                    <YAxis
                      dataKey="avgMetric"
                      type="number"
                      tick={{ fill: '#9CA3AF' }}
                      tickLine={{ stroke: '#4B5563' }}
                      axisLine={{ stroke: '#4B5563' }}
                      label={{
                        value: 'Average Metric',
                        angle: -90,
                        position: 'left',
                        offset: 0,
                        style: { fill: '#9CA3AF', fontSize: 12 },
                      }}
                    />
                    <Tooltip
                      content={CustomTooltip(xKey, xLabel)}
                      cursor={{ stroke: '#6B7280', strokeDasharray: '3 3' }}
                    />
                    <Scatter
                      data={data}
                      fill="#6366f1"
                      fillOpacity={0.8}
                      stroke="#818cf8"
                      strokeWidth={1}
                      r={6}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </ChartContainer>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

PlotSection.displayName = 'PlotSection';

export default PlotSection;
