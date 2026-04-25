export type LatLng = {
  lat: number;
  lon: number;
};

export type RouteStep = {
  instruction: string;
  end: LatLng;
};

export type HazardWarning = {
  message: string;
  level: "urgent" | "normal";
  ts: number;
};
