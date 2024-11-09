import LeaderboardPage from '@/components/LeaderBoardPage';
import { fetchSuite, fetchModelList, fetchModelData } from './utils/fetchData';

export default async function Home() {
  const suite = await fetchSuite();
  const model_names = await fetchModelList();
  const models = await Promise.all(model_names.map(fetchModelData));

  return <LeaderboardPage models={models} suite={suite} />;
}
