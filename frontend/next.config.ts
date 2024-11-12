import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'export',
  experimental: {
    optimizeCss: true,
  },
};

export default nextConfig;
