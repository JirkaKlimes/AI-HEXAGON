import { fetchSuite, fetchModelList, fetchModelData } from './utils/fetchData';
import HomePage from '@/components/pages/home';

export default async function Home() {
  const suite = await fetchSuite();
  const model_names = await fetchModelList();
  const models = await Promise.all(model_names.map(fetchModelData));

  return <HomePage models={models} suite={suite} />;
}
