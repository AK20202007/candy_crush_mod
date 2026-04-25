import type { LatLng, RouteStep } from "../types";

function stripHtml(text: string): string {
  return text.replace(/<[^>]+>/g, "").trim();
}

type GoogleDirectionsResponse = {
  status?: string;
  error_message?: string;
  routes?: Array<{
    legs?: Array<{
      steps?: Array<{
        html_instructions?: string;
        end_location?: { lat?: number; lng?: number };
      }>;
    }>;
  }>;
};

export async function fetchGoogleWalkingSteps(
  apiKey: string,
  origin: LatLng,
  destinationText: string
): Promise<RouteStep[]> {
  const key = apiKey.trim();
  const destination = destinationText.trim();
  if (!key) {
    throw new Error("Google Maps API key is required.");
  }
  if (!destination) {
    throw new Error("Destination is required.");
  }

  const originParam = encodeURIComponent(`${origin.lat},${origin.lon}`);
  const destinationParam = encodeURIComponent(destination);
  const keyParam = encodeURIComponent(key);
  const url =
    "https://maps.googleapis.com/maps/api/directions/json" +
    `?origin=${originParam}&destination=${destinationParam}&mode=walking&units=metric&key=${keyParam}`;

  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Google Directions HTTP ${resp.status}`);
  }
  const data = (await resp.json()) as GoogleDirectionsResponse;
  if (data.status !== "OK") {
    throw new Error(`Google Directions failed: ${data.status ?? "unknown"} ${data.error_message ?? ""}`.trim());
  }

  const legs = data.routes?.[0]?.legs ?? [];
  const stepsRaw = legs[0]?.steps ?? [];
  const steps: RouteStep[] = [];
  for (const s of stepsRaw) {
    const lat = s.end_location?.lat;
    const lon = s.end_location?.lng;
    const instruction = stripHtml(s.html_instructions ?? "");
    if (instruction && typeof lat === "number" && typeof lon === "number") {
      steps.push({ instruction, end: { lat, lon } });
    }
  }
  if (steps.length === 0) {
    throw new Error("Route has no usable steps.");
  }
  return steps;
}
