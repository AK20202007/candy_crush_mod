import type { LatLng, RouteStep } from "./types";

const EARTH_RADIUS_M = 6371000;

function toRad(value: number): number {
  return (value * Math.PI) / 180;
}

export function haversineMeters(a: LatLng, b: LatLng): number {
  const p1 = toRad(a.lat);
  const p2 = toRad(b.lat);
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lon - a.lon);
  const aa =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(p1) * Math.cos(p2) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.atan2(Math.sqrt(aa), Math.sqrt(1 - aa));
}

export type ProgressResult = {
  nextIndex: number;
  confirmedHits: number;
  reachedStep: boolean;
  reachedDestination: boolean;
  distanceToTargetM: number;
};

export function progressNavigation(
  steps: RouteStep[],
  currentIndex: number,
  location: LatLng,
  arrivalRadiusM: number,
  confirmedHits: number,
  confirmHitsNeeded: number
): ProgressResult {
  if (steps.length === 0 || currentIndex >= steps.length) {
    return {
      nextIndex: currentIndex,
      confirmedHits: 0,
      reachedStep: false,
      reachedDestination: true,
      distanceToTargetM: 0
    };
  }

  const target = steps[currentIndex].end;
  const distanceToTargetM = haversineMeters(location, target);
  const inRadius = distanceToTargetM <= arrivalRadiusM;
  const hits = inRadius ? confirmedHits + 1 : 0;
  const reachedStep = hits >= confirmHitsNeeded;
  if (!reachedStep) {
    return {
      nextIndex: currentIndex,
      confirmedHits: hits,
      reachedStep: false,
      reachedDestination: false,
      distanceToTargetM
    };
  }

  const nextIndex = currentIndex + 1;
  return {
    nextIndex,
    confirmedHits: 0,
    reachedStep: true,
    reachedDestination: nextIndex >= steps.length,
    distanceToTargetM
  };
}
