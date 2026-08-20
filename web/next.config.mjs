/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow large document uploads through the API route
  experimental: {
    serverActions: {
      bodySizeLimit: '25mb',
    },
  },
};

export default nextConfig;
