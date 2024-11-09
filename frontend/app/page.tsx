import LeaderboardPage from '@/components/LeaderBoardPage';
import { fetchSuite, fetchModelList, fetchModelData } from './utils/fetchData';
import ModelAnalysisCharts from '@/components/ModelAnalysisCharts';
import ModelCard from '@/components/ModelCard';

export default async function Home() {
  const suite = await fetchSuite();
  const model_names = await fetchModelList();
  const models = await Promise.all(model_names.map(fetchModelData));

  return <LeaderboardPage models={models} suite={suite} />;

  //     <div className="dark:bg-gray-900 min-h-screen text-white">
  //       <main className="container mx-auto py-12 px-4">
  //         <div className="space-y-8">
  //           <ModelAnalysisCharts models={models} suite={suite} />
  //           <div className="space-y-2">
  //             <h1 className="text-3xl font-bold tracking-tight">Model Cards</h1>
  //           </div>

  //           <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
  //             {models.map((model) => (
  //               <div key={model.name} className="flex">
  //                 <ModelCard model={model} suite={suite} />
  //               </div>
  //             ))}
  //           </div>
  //         </div>
  //       </main>
  //     </div>
  //   );
}
