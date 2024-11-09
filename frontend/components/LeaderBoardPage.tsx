import React from 'react';
import Link from 'next/link';
import { SiGithub } from 'react-icons/si';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ModelData, Suite } from '@/app/utils/types';
import ModelAnalysisCharts from '@/components/ModelAnalysisCharts';
import ModelCard from '@/components/ModelCard';
import { Alert, AlertDescription } from './ui/alert';

interface LeaderboardPageProps {
  models: ModelData[];
  suite: Suite;
}

const LeaderboardPage: React.FC<LeaderboardPageProps> = ({ models, suite }) => {
  // Sort models by their best performing variation
  const sortedModels = [...models].sort((a, b) => {
    const aMetrics = Object.values(
      Object.values(a.variations)[0].metrics
    ) as number[];
    const bMetrics = Object.values(
      Object.values(b.variations)[0].metrics
    ) as number[];

    const aAvg = aMetrics.reduce((sum, val) => sum + val, 0) / aMetrics.length;
    const bAvg = bMetrics.reduce((sum, val) => sum + val, 0) / bMetrics.length;

    return bAvg - aAvg;
  });

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header Section */}
      <div className="border-b border-gray-800">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-teal-400">
              AI-HEXAGON Leaderboard
            </h1>
            <Button
              variant="outline"
              size="lg"
              className="flex items-center gap-2"
              asChild
            >
              <Link
                href="https://github.com/JirkaKlimes/AI-HEXAGON"
                target="_blank"
                rel="noopener noreferrer"
              >
                <SiGithub className="w-5 h-5" />
                View on GitHub
              </Link>
            </Button>
          </div>

          <div className="space-y-4">
            <p className="text-lg text-gray-300">
              An objective benchmarking framework for evaluating neural network
              architectures independently of natural language processing tasks.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-sm text-gray-400">
                    Current Suite
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xl font-bold">{suite.name}</p>
                </CardContent>
              </Card>

              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-sm text-gray-400">
                    Models Evaluated
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xl font-bold">{models.length}</p>
                </CardContent>
              </Card>

              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-sm text-gray-400">
                    Sequence Length
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xl font-bold">{suite.sequence_length}</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Charts Section */}
      <section className="container mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mb-6">Performance Analysis</h2>
        <ModelAnalysisCharts models={sortedModels} suite={suite} />
      </section>

      {/* Leaderboard Section */}
      <section className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Model Leaderboard</h2>
          <Alert className="w-auto bg-gray-900 border-gray-800">
            <AlertDescription>
              All models tested with {suite.sequence_length} sequence length and{' '}
              {suite.vocab_size} vocabulary size
            </AlertDescription>
          </Alert>
        </div>

        <div className="grid grid-cols-1 gap-6">
          {sortedModels.map((model, index) => (
            <div key={model.name} className="relative">
              <div className="absolute -left-8 top-4 text-2xl font-bold text-gray-600">
                #{index + 1}
              </div>
              <ModelCard model={model} suite={suite} />
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12">
        <div className="container mx-auto px-4 py-8">
          <p className="text-center text-gray-400">
            AI-HEXAGON is an open-source project. Contribute on{' '}
            <Link
              href="https://github.com/JirkaKlimes/AI-HEXAGON"
              className="text-blue-400 hover:text-blue-300"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </Link>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LeaderboardPage;
