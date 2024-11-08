import { Suite, ModelResult, ModelData } from './types';

const repo = `${process.env.GITHUB_REPOSITORY_OWNER}/${process.env.GITHUB_REPOSITORY}`;

async function fetchFile<T>(path: string): Promise<T> {
  const url = `https://raw.githubusercontent.com/${repo}/main/${path}`;
  console.log(url);
  const response = await fetch(url, { next: { revalidate: 600 } });
  if (!response.ok) {
    throw new Error(`Failed to fetch file: ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchSuite(): Promise<Suite> {
  return fetchFile<Suite>('results/suite.json');
}

export async function fetchModelList(): Promise<string[]> {
  const url = `https://api.github.com/repos/${repo}/contents/results`;
  const response = await fetch(url, { next: { revalidate: 600 } });
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`);
  }
  const contents = await response.json();
  const folders = contents.filter((file: any) => file.type === 'dir');
  return folders.map((folder: any) => folder.name);
}

export async function fetchModelData(name: string): Promise<ModelData> {
  const path = `results/${name}/model.result.json`;
  var f = await fetchFile<ModelResult>(path);
  const result: ModelData = {
    ...f,
    source: `https://github.com/${repo}/blob/main/results/${name}/model.py`,
  };
  return result;
}
