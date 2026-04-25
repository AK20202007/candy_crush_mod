import { requireOptionalNativeModule } from "expo-modules-core";
import type { HazardWarning } from "../types";

type VisionNativeModule = {
  start: (config: Record<string, unknown>) => Promise<void>;
  stop: () => Promise<void>;
  addListener?: (eventName: string, listener: (payload: unknown) => void) => { remove: () => void };
};

const native = requireOptionalNativeModule<VisionNativeModule>("LAHacksVision");

export const hasNativeVision = Boolean(native?.start);

export async function startVision(config: Record<string, unknown>): Promise<void> {
  if (!native) {
    return;
  }
  await native.start({
    obstacleAreaRatio: 0.12,
    personCenterRadius: 0.22,
    warningCooldownS: 2.5,
    confirmFrames: 2,
    ...config
  });
}

export async function stopVision(): Promise<void> {
  if (!native) {
    return;
  }
  await native.stop();
}

/**
 * Prefer the Expo module's `addListener` (from `Events()` in Swift) instead of `NativeEventEmitter`,
 * which expects legacy `RCTEventEmitter` shape and can throw or misbehave with Expo Modules.
 */
export function subscribeWarnings(onWarning: (warning: HazardWarning) => void): () => void {
  if (!native || typeof native.addListener !== "function") {
    return () => undefined;
  }
  try {
    const sub = native.addListener("visionWarning", (payload: unknown) => {
      const obj = payload as { message?: string; level?: "urgent" | "normal"; ts?: number };
      if (!obj?.message) {
        return;
      }
      onWarning({
        message: obj.message,
        level: obj.level === "normal" ? "normal" : "urgent",
        ts: typeof obj.ts === "number" ? obj.ts : Date.now()
      });
    });
    return () => {
      sub.remove();
    };
  } catch {
    return () => undefined;
  }
}
