import Constants from "expo-constants";

type ExtraConfig = {
  navApiBaseUrl?: string;
};

export function normalizeApiBaseUrl(value: string | undefined | null): string {
  return (value ?? "").trim().replace(/\/+$/, "");
}

export function getNavApiBaseUrl(): string {
  const extra = Constants.expoConfig?.extra as ExtraConfig | undefined;
  return normalizeApiBaseUrl(extra?.navApiBaseUrl);
}

export function formatNavApiStatus(baseUrl: string): string {
  if (!baseUrl) {
    return "cloud off";
  }
  const host = baseUrl.replace(/^[a-z][a-z0-9+.-]*:\/\//i, "").split("/")[0];
  return host ? `cloud ${host}` : "cloud configured";
}
