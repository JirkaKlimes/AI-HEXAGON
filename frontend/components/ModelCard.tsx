'use client';

import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';

// Mock data for radar charts - you'll replace this with actual data
const generateRandomData = () => {
  return [
    { subject: 'Reasoning', value: Math.random() * 100 },
    { subject: 'Knowledge', value: Math.random() * 100 },
    { subject: 'Math', value: Math.random() * 100 },
    { subject: 'Coding', value: Math.random() * 100 },
    { subject: 'Vision', value: Math.random() * 100 },
    { subject: 'Safety', value: Math.random() * 100 },
  ];
};

interface ModelCardProps {
  name: string;
}

export function ModelCard({ name }: ModelCardProps) {
  const data = generateRandomData();

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg">{name}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={data}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <Radar
                name={name}
                dataKey="value"
                stroke="#2563eb"
                fill="#3b82f6"
                fillOpacity={0.6}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
