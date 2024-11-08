import PlotSection from '@/components/PlotSection';
import { fetchSuite, fetchModelList, fetchModelData } from './utils/fetchData';
import ModelCards from '@/components/ModelCards';

export default async function Home() {
  const suite = await fetchSuite();
  const models = await fetchModelList();
  const results = await Promise.all(models.map(fetchModelData));

  return (
    <div className="dark:bg-gray-900 min-h-screen text-white">
      <header className="text-center py-8">
        <h1 className="text-4xl font-bold">AI-HEXAGON</h1>
        <p className="mt-4">
          AI-HEXAGON is an objective benchmarking framework designed to evaluate
          neural network architectures independently of natural language
          processing tasks.
        </p>
      </header>
      <PlotSection results={results} suite={suite} />
      <ModelCards results={results} suite={suite} />
    </div>
  );
}
