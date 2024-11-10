'use client';

import React, { useState } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { SiGithub, SiArxiv } from 'react-icons/si';
import {
  RadarChart,
  PolarGrid,
  Radar,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
} from 'recharts';
import { ModelData, ModelVariationResult, Suite } from '@/app/utils/types';

interface ModelCardProps {
  model: ModelData;
  suite: Suite;
  rank: number;
  avgMetric: number;
  bestSize: number;
  bestCompute: number;
}

interface RadarDataPoint {
  metric: string;
  value: number;
  description: string;
}

interface ModelVariationContentProps {
  variation: ModelVariationResult;
  metricDescriptions: Record<string, string>;
  sequenceLength: number;
  sequenceLengths: number[];
}

const formatToMB = (bytes: number) => (bytes / (1024 * 1024)).toFixed(2);
const formatToGFLOPs = (flops: number) => (flops / 1e9).toFixed(2);
const formatToPercentage = (decimal: number) =>
  `${(decimal * 100).toFixed(1)}%`;

const CustomTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ value: number; payload: RadarDataPoint }>;
}) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-gray-800 p-3 rounded-lg border border-gray-700 shadow-lg">
        <p className="text-sm font-medium text-gray-200">{data.metric}</p>
        <p className="text-sm text-gray-400 mt-1">{data.description}</p>
        <p className="text-sm font-medium text-blue-400 mt-2">
          {formatToPercentage(data.value / 100)}
        </p>
      </div>
    );
  }
  return null;
};

const ModelVariationContent = ({
  variation,
  metricDescriptions,
  sequenceLength,
  sequenceLengths,
}: ModelVariationContentProps) => {
  const radarData: RadarDataPoint[] = Object.entries(variation.metrics).map(
    ([name, value]) => ({
      metric: name,
      value: value * 100,
      description: metricDescriptions[name],
    })
  );

  return (
    <div className="space-y-6">
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke="#374151" />
            <RechartsTooltip content={<CustomTooltip />} />
            <Radar
              name="Metrics"
              dataKey="value"
              stroke="#60A5FA"
              fill="#60A5FA"
              fillOpacity={0.3}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-400">Model Size</p>
          <p className="text-2xl font-bold text-gray-100">
            {formatToMB(variation.model_stats.size)} MB
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Doubling rate: {variation.model_stats.size_doubling_rate.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {variation.model_stats.size_big_o}
          </p>
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-400">
            Compute Requirements
          </p>
          <p className="text-2xl font-bold text-gray-100">
            {formatToGFLOPs(variation.model_stats.flops)} GFLOPs
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Doubling rate:{' '}
            {variation.model_stats.flops_doubling_rate.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {variation.model_stats.flops_big_o}
          </p>
        </div>
      </div>

      <div className="text-sm text-gray-500 space-y-1">
        <p>* Size and GFLOPs computed at sequence length: {sequenceLength}</p>
        <p>
          * Doubling rates computed across sequence lengths:{' '}
          {sequenceLengths.join(' • ')}
        </p>
      </div>
    </div>
  );
};

export const ModelCard = ({
  model,
  suite,
  rank,
  avgMetric,
  bestSize,
  bestCompute,
}: ModelCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const metricDescriptions = Object.fromEntries(
    suite.metrics.map((metric) => [metric.name, metric.description])
  );

  const variations = Object.entries(model.variations);
  const [selectedVariation, setSelectedVariation] = useState(variations[0][0]);

  return (
    <div
      className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden cursor-pointer transition-all duration-200"
      onClick={() => setIsExpanded(!isExpanded)}
    >
      {/* Condensed View */}
      <div className="p-4 flex items-center gap-6">
        <div className="w-12 text-2xl font-bold text-gray-500">#{rank}</div>
        <div className="flex-1">
          <h3 className="font-semibold text-lg">{model.title}</h3>
        </div>
        <div className="flex gap-8">
          <div>
            <p className="text-sm text-gray-400">Size</p>
            <p className="font-medium">{formatToMB(bestSize)} MB</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Compute</p>
            <p className="font-medium">{formatToGFLOPs(bestCompute)} GFLOPs</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Avg. Accuracy</p>
            <p className="font-medium">{formatToPercentage(avgMetric)}</p>
          </div>
        </div>
      </div>

      {/* Expanded View */}
      {isExpanded && (
        <div
          className="p-6 border-t border-gray-800"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              {variations.length > 1 && (
                <TooltipProvider>
                  <Select
                    value={selectedVariation}
                    onValueChange={setSelectedVariation}
                  >
                    <SelectTrigger className="w-32 bg-gray-800 border-gray-700 h-8 text-sm">
                      <SelectValue placeholder="Select variant" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-800 border border-gray-700 text-gray-300">
                      {variations.map(([key, variation]) => (
                        <Tooltip key={key}>
                          <TooltipTrigger asChild>
                            <div>
                              <SelectItem
                                value={key}
                                className="hover:bg-gray-700 focus:bg-gray-700 text-gray-300"
                              >
                                {key.charAt(0).toUpperCase() + key.slice(1)}
                              </SelectItem>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent
                            side="right"
                            className="w-80 bg-gray-800 border border-gray-700 text-gray-300"
                          >
                            <div className="space-y-2">
                              <h4 className="text-sm font-semibold text-gray-200">
                                Arguments
                              </h4>
                              <pre className="text-xs bg-gray-700 p-2 rounded-md overflow-auto">
                                {JSON.stringify(variation.arguments, null, 2)}
                              </pre>
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      ))}
                    </SelectContent>
                  </Select>
                </TooltipProvider>
              )}
            </div>
            <div className="flex items-center gap-2">
              {model.paper && (
                <a
                  href={model.paper}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 hover:bg-gray-800 rounded-full transition-colors"
                  onClick={(e) => e.stopPropagation()}
                >
                  <SiArxiv className="w-5 h-5 text-gray-400 hover:text-gray-300" />
                </a>
              )}
              <a
                href={model.source}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 hover:bg-gray-800 rounded-full transition-colors"
                onClick={(e) => e.stopPropagation()}
              >
                <SiGithub className="w-5 h-5 text-gray-400 hover:text-gray-300" />
              </a>
            </div>
          </div>

          {/* Metrics and Stats */}
          <ModelVariationContent
            variation={model.variations[selectedVariation]}
            metricDescriptions={metricDescriptions}
            sequenceLength={suite.sequence_length}
            sequenceLengths={suite.sequence_lengths}
          />

          {/* Description and Authors */}
          <div className="bg-gray-800 p-4 rounded-lg space-y-4 mt-6">
            <div>
              <h4 className="font-medium text-gray-300 mb-1">Description</h4>
              <p className="text-sm text-gray-400">{model.description}</p>
            </div>

            {model.authors && (
              <div>
                <h4 className="font-medium text-gray-300 mb-1">Authors</h4>
                <p className="text-sm text-gray-400">
                  {model.authors.join(', ')}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelCard;
