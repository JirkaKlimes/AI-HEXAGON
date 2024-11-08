export interface Suite {
  name: string;
  description: string;
  vocab_size: number;
  sequence_length: number;
  sequence_lengths: number[];
  metrics: Metric[];
}

export interface Metric {
  name: string;
  description: string;
  tests: TestEntry[];
}

export interface TestEntry {
  weight: number;
  test: Test;
}

export interface Test {
  name: string;
}

export interface ModelResult {
  title: string;
  description: string;
  authors: string[] | null;
  paper: string | null;
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

export interface ModelData extends ModelResult {
  source: string;
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
