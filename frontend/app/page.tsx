import { ModelCard } from '@/components/ModelCard';

interface ModelResult {
  name: string;
  description: string;
  metrics: { [key: string]: number };
  model_stats: {
    size: number;
    size_doubling_rate: number;
    size_big_o: string;
    flops: number;
    flops_doubling_rate: number;
    flops_big_o: string;
  };
}

async function fetchFile(path: string) {
  const repo = process.env.GITHUB_REPO;
  if (!repo) {
    throw new Error('Missing GITHUB_REPO environment variable');
  }
  const url = `https://raw.githubusercontent.com/${repo}/refs/heads/main/${path}`;
  const response = await fetch(url, { next: { revalidate: 600 } });
  if (!response.ok) {
    throw new Error(`Failed to fetch suite: ${response.statusText}`);
  }
  return response.json();
}

async function listModels() {
  const repo = process.env.GITHUB_REPO;
  if (!repo) {
    throw new Error('Missing GITHUB_REPO environment variable');
  }
  const url = `https://api.github.com/repos/${repo}/contents/results`;
  const response = await fetch(url, { next: { revalidate: 600 } });
  if (!response.ok) {
    throw new Error(`Failed to fetch suite: ${response.statusText}`);
  }
  const contents = await response.json();
  const folders = contents.filter((file: any) => file.type === 'dir');
  const models = folders.map((folder: any) => folder.name);
  return models;
}

export default async function Home() {
  const suite = await fetchFile('./results/suite.json');
  const models = await listModels();

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        {/* Header Section */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            {suite.name}
          </h1>
          <p className="text-xl text-gray-600">{suite.description}</p>
        </div>

        {/* Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((model: string) => (
            <ModelCard key={model} name={model} />
          ))}
        </div>
      </div>
    </div>
  );
}
