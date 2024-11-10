'use client';

import { TooltipProvider } from '@/components/ui/tooltip';
import { Hexagon } from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useState } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ModelData, ModelVariationResult, Suite } from '@/app/utils/types';
import ModelCard from '../ModelCard';

interface ProcessedModelData {
  name: string;
  accuracy: number;
  size: number;
  compute: number;
  size_doubling_rate: number;
  compute_doubling_rate: number;
}

interface HomePageProps {
  models: ModelData[];
  suite: Suite;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    payload: ProcessedModelData;
  }>;
}

interface BestResult {
  variation: ModelVariationResult;
  avgMetric: number;
}

type ViewMode = 'performance' | 'scaling';

const formatToMB = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)}`;
const formatToGFLOPs = (flops: number) => `${(flops / 1e9).toFixed(1)}`;
const formatToPercentage = (decimal: number) =>
  `${(decimal * 100).toFixed(1)}%`;

const HomePage = ({ models, suite }: HomePageProps) => {
  const [viewMode, setViewMode] = useState<ViewMode>('performance');

  // Process models data to select the best variation per model
  const processedModels: ProcessedModelData[] = models.map((model) => {
    const variations = Object.values(model.variations);

    const bestVariation = variations.reduce<BestResult>(
      (best, current) => {
        const currentMetrics = Object.values(current.metrics);
        const currentAvg =
          currentMetrics.length > 0
            ? currentMetrics.reduce((a, b) => a + b, 0) / currentMetrics.length
            : 0;

        if (!best.variation || currentAvg > best.avgMetric) {
          return {
            variation: current,
            avgMetric: currentAvg,
          };
        }
        return best;
      },
      { variation: variations[0], avgMetric: -Infinity }
    );

    return {
      name: model.title,
      accuracy: bestVariation.avgMetric,
      size: bestVariation.variation.model_stats.size,
      compute: bestVariation.variation.model_stats.flops,
      size_doubling_rate:
        bestVariation.variation.model_stats.size_doubling_rate,
      compute_doubling_rate:
        bestVariation.variation.model_stats.flops_doubling_rate,
    };
  });

  const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
          <p className="text-gray-200">{`Model: ${data.name}`}</p>
          <p className="text-blue-400">{`Accuracy: ${formatToPercentage(
            data.accuracy
          )}`}</p>
          <p className="text-green-400">
            {viewMode === 'performance'
              ? payload[0].name === 'Size'
                ? `Size: ${formatToMB(data.size)} MB`
                : `Compute: ${formatToGFLOPs(data.compute)} GFLOPs`
              : `${payload[0].name} Rate: ${payload[0].value.toFixed(2)}`}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-black text-gray-100">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-2">
            <Hexagon className="h-8 w-8 text-blue-500" />
            <h1 className="text-2xl font-bold text-gray-100">AI-HEXAGON</h1>
          </div>
        </div>
        <Separator className="my-6 bg-gray-800" />

        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-semibold">
              Neural Network Performance Analysis
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              {viewMode === 'performance'
                ? `Measured at sequence length: ${suite.sequence_length}`
                : `Tested on sequence lengths: ${suite.sequence_lengths.join(
                    ' • '
                  )}`}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-gray-400 text-sm">View:</span>
            <Select
              value={viewMode}
              onValueChange={(value: ViewMode) => setViewMode(value)}
            >
              <SelectTrigger className="w-32 bg-gray-800 border-gray-700 h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="performance">Performance</SelectItem>
                <SelectItem value="scaling">Scaling</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Size Chart */}
          <div className="bg-gray-900 p-6 rounded-xl">
            <div className="mb-4">
              <h2 className="text-xl font-semibold">
                {viewMode === 'performance'
                  ? 'Model Size'
                  : 'Size Scaling Rate'}
              </h2>
              {viewMode === 'performance' && (
                <p className="text-sm text-gray-400 mt-1">
                  Measured in Megabytes (MB)
                </p>
              )}
            </div>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 20, right: 30, bottom: 20, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey={
                      viewMode === 'performance' ? 'size' : 'size_doubling_rate'
                    }
                    name="Size"
                    type="number"
                    stroke="#9CA3AF"
                    tickFormatter={
                      viewMode === 'performance' ? formatToMB : undefined
                    }
                  />
                  <YAxis
                    dataKey="accuracy"
                    name="Accuracy"
                    stroke="#9CA3AF"
                    tickFormatter={formatToPercentage}
                    domain={[0, 1]}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Scatter name="Size" data={processedModels} fill="#60A5FA" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Compute Chart */}
          <div className="bg-gray-900 p-6 rounded-xl">
            <div className="mb-4">
              <h2 className="text-xl font-semibold">
                {viewMode === 'performance'
                  ? 'Model Compute'
                  : 'Compute Scaling Rate'}
              </h2>
              {viewMode === 'performance' && (
                <p className="text-sm text-gray-400 mt-1">
                  Measured in GigaFLOPs
                </p>
              )}
            </div>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 20, right: 30, bottom: 20, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey={
                      viewMode === 'performance'
                        ? 'compute'
                        : 'compute_doubling_rate'
                    }
                    name="Compute"
                    type="number"
                    stroke="#9CA3AF"
                    tickFormatter={
                      viewMode === 'performance' ? formatToGFLOPs : undefined
                    }
                  />
                  <YAxis
                    dataKey="accuracy"
                    name="Accuracy"
                    stroke="#9CA3AF"
                    tickFormatter={formatToPercentage}
                    domain={[0, 1]}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Scatter
                    name="Compute"
                    data={processedModels}
                    fill="#34D399"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Leaderboard Section */}
        <div className="mt-12">
          <h2 className="text-2xl font-semibold mb-6">Model Leaderboard</h2>
          <TooltipProvider>
            <div className="space-y-2">
              {processedModels
                .sort((a, b) => b.accuracy - a.accuracy)
                .map((model, index) => (
                  <ModelCard
                    key={model.name}
                    model={models.find((m) => m.title === model.name)!}
                    suite={suite}
                    rank={index + 1}
                    avgMetric={model.accuracy}
                    bestSize={model.size}
                    bestCompute={model.compute}
                  />
                ))}
            </div>
          </TooltipProvider>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
