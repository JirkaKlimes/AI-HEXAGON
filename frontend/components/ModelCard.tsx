'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';
import { ModelData, ModelVariationResult, Suite } from '@/app/utils/types';

const bytesToMB = (bytes: number) => (bytes / (1024 * 1024)).toFixed(2);
const flopsToGLOPs = (flops: number) =>
  (flops / (1000 * 1000 * 1000)).toFixed(2);

interface ModelCardProps {
  model: ModelData;
  suite: Suite;
}

const ModelVariationContent = ({
  variation,
  metricDescriptions,
  sequenceLength,
  sequenceLengths,
}: {
  variation: ModelVariationResult;
  metricDescriptions: Record<string, string>;
  sequenceLength: number;
  sequenceLengths: number[];
}) => {
  const radarData = Object.entries(variation.metrics).map(([name, value]) => ({
    metric: name,
    value: value * 100,
    description: metricDescriptions[name],
  }));

  return (
    <div className="space-y-6">
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" />
            <Radar
              name="Metrics"
              dataKey="value"
              stroke="#2563eb"
              fill="#2563eb"
              fillOpacity={0.3}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm font-medium">Model Size</p>
          <p className="text-2xl font-bold">
            {bytesToMB(variation.model_stats.size)} MB
          </p>
          <p className="text-sm text-muted-foreground">
            Doubling rate: {variation.model_stats.size_doubling_rate.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-sm font-medium">Compute Requirements</p>
          <p className="text-2xl font-bold">
            {flopsToGLOPs(variation.model_stats.flops)} GLOPs
          </p>
          <p className="text-sm text-muted-foreground">
            Doubling rate:{' '}
            {variation.model_stats.flops_doubling_rate.toFixed(2)}
          </p>
        </div>
      </div>

      <div className="text-sm text-muted-foreground">
        <p>* Size and GLOPs computed at sequence length: {sequenceLength}</p>
        <p>
          * Doubling rates computed across sequence lengths:{' '}
          {sequenceLengths.join(', ')}
        </p>
      </div>
    </div>
  );
};

export const ModelCard = ({ model, suite }: ModelCardProps) => {
  const metricDescriptions = Object.fromEntries(
    suite.metrics.map((metric) => [metric.name, metric.description])
  );

  const variations = Object.entries(model.variations);
  const [selectedVariation, setSelectedVariation] = useState(variations[0][0]);

  return (
    <Card className="w-full bg-gradient-to-b from-gray-800 to-gray-900">
      <CardHeader className="space-y-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <CardTitle>{model.title}</CardTitle>
            {variations.length > 1 && (
              <TooltipProvider>
                <Select
                  value={selectedVariation}
                  onValueChange={setSelectedVariation}
                >
                  <SelectTrigger className="w-[120px]">
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
                            <h4 className="text-sm font-semibold text-white">
                              Arguments
                            </h4>
                            <pre className="text-xs bg-gray-700 p-2 rounded-md overflow-auto text-gray-300">
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
                className="p-2 hover:bg-secondary rounded-full transition-colors"
              >
                <SiArxiv className="w-5 h-5" />
              </a>
            )}
            <a
              href={model.source}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-secondary rounded-full transition-colors"
            >
              <SiGithub className="w-5 h-5" />
            </a>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <ModelVariationContent
          variation={model.variations[selectedVariation]}
          metricDescriptions={metricDescriptions}
          sequenceLength={suite.sequence_length}
          sequenceLengths={suite.sequence_lengths}
        />

        <div className="bg-gray-800 p-4 rounded-md space-y-4">
          <div>
            <h3 className="font-medium text-white">Description</h3>
            <p className="text-sm text-gray-300">{model.description}</p>
          </div>

          {model.authors && (
            <div>
              <h3 className="font-medium text-white">Authors</h3>
              <p className="text-sm text-gray-300">
                {model.authors.join(', ')}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelCard;
