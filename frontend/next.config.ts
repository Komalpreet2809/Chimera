import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server's origin is localhost, so Next 16 blocks its dev-only
  // resources (client bundle, HMR, fonts) when the app is opened via
  // 127.0.0.1 — which silently breaks hydration and makes every button dead.
  // Allow the loopback IP so both hostnames work in development.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
