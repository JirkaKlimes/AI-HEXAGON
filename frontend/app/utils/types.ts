export interface Test {
  name: string;
}

export interface WeightedTest {
  weight: number;
  test: Test;
}

export interface Metric {
  name: string;
  description: string;
  tests: WeightedTest[];
}

export interface ModelStats {
  size: number;
  size_doubling_rate: number;
  size_big_o: string;
  flops: number;
  flops_doubling_rate: number;
  flops_big_o: string;
}

export interface ModelVariationResult {
  arguments: Record<string, any>;
  metrics: Record<string, number>;
  model_stats: ModelStats;
}

export interface ModelResult {
  name: string;
  title: string;
  description: string;
  authors: string[] | null;
  paper: string | null;
  variations: Record<string, ModelVariationResult>;
}
export interface ModelData extends ModelResult {
  source: string;
}

export interface Suite {
  name: string;
  description: string;
  vocab_size: number;
  sequence_length: number;
  sequence_lengths: number[];
  metrics: Metric[];
}

export interface GitHubContentLinks {
  self: string;
  git: string;
  html: string;
}

export interface GitHubContentItem {
  name: string;
  path: string;
  sha: string;
  size: number;
  url: string;
  html_url: string;
  git_url: string;
  download_url: string | null;
  type: 'file' | 'dir';
  _links: GitHubContentLinks;
}

export type GitHubContentsResponse = GitHubContentItem[];
