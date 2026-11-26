import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.xauusd.analyzer",
  appName: "XAUUSD Analyzer",
  webDir: "dist",
  server: {
    androidScheme: "https",
  },
  android: {
    backgroundColor: "#F6F7F9",
    allowMixedContent: false,
  },
};

export default config;
