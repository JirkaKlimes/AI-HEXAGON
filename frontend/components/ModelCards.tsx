'use client';

import React, { FC, ReactNode } from 'react';
import { ModelData, Suite } from '@/app/utils/types';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts';
import { ExternalLinkIcon } from 'lucide-react';

interface ModelCardsProps {
  results: ModelData[];
  suite: Suite;
}

const CardContainer: FC<{ children: ReactNode }> = ({ children }) => (
  <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
    {children}
  </div>
);

const CustomRadarTooltip: React.FC<TooltipProps<number, string>> = ({
  active,
  payload,
}) => {
  if (active && payload && payload.length > 0) {
    const dataPoint = payload[0].payload as { metric: string; value: number };
    return (
      <div className="bg-gray-800 shadow-xl p-3 rounded-lg border border-gray-700">
        <p className="font-medium text-white">{dataPoint.metric}</p>
        <p className="text-gray-300">Value: {dataPoint.value.toFixed(3)}</p>
      </div>
    );
  }
  return null;
};

const ModelCards: FC<ModelCardsProps> = ({ results }) => {
  return (
    <section className="w-full max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="space-y-16">
        <div>
          <h3 className="text-2xl font-semibold text-white mb-6">Models</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {results.map((model) => {
              const data = Object.entries(model.metrics).map(
                ([metric, value]) => ({
                  metric,
                  value,
                })
              );

              return (
                <CardContainer key={model.title}>
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-medium text-gray-200">
                      {model.title}
                    </h4>
                    {model.source && (
                      <a
                        href={model.source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-gray-400 hover:text-gray-200 flex items-center"
                      >
                        <ExternalLinkIcon className="h-5 w-5 inline-block" />
                      </a>
                    )}
                  </div>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={data}>
                      <PolarGrid stroke="#374151" />
                      <PolarAngleAxis dataKey="metric" stroke="#9CA3AF" />
                      <PolarRadiusAxis stroke="#4B5563" />
                      <Radar
                        name={model.title}
                        dataKey="value"
                        stroke="#818cf8"
                        fill="#6366f1"
                        fillOpacity={0.8}
                      />
                      <Tooltip content={<CustomRadarTooltip />} />
                    </RadarChart>
                  </ResponsiveContainer>
                  <p className="mt-4 text-gray-300">{model.description}</p>
                  {model.paper && (
                    <p className="mt-2">
                      <a
                        href={model.paper}
                        className="text-blue-400 underline"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Read the paper
                      </a>
                    </p>
                  )}
                  {model.authors && (
                    <p className="mt-2 text-gray-300">
                      Authors: {model.authors.join(', ')}
                    </p>
                  )}
                  <div className="mt-4">
                    <h4 className="font-semibold text-gray-200">
                      Model Stats:
                    </h4>
                    <ul className="list-disc list-inside text-gray-300">
                      <li>
                        Size:{' '}
                        {(model.model_stats.size / 1e6).toLocaleString(
                          undefined,
                          {
                            maximumFractionDigits: 2,
                          }
                        )}{' '}
                        MB
                      </li>
                      <li>
                        Size Doubling Rate:{' '}
                        {model.model_stats.size_doubling_rate.toLocaleString(
                          undefined,
                          { maximumFractionDigits: 2 }
                        )}
                      </li>
                      <li>
                        FLOPs:{' '}
                        {(model.model_stats.flops / 1e9).toLocaleString(
                          undefined,
                          {
                            maximumFractionDigits: 2,
                          }
                        )}{' '}
                        GFLOPs
                      </li>
                      <li>
                        FLOPs Doubling Rate:{' '}
                        {model.model_stats.flops_doubling_rate.toLocaleString(
                          undefined,
                          { maximumFractionDigits: 2 }
                        )}
                      </li>
                    </ul>
                  </div>
                </CardContainer>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelCards;
