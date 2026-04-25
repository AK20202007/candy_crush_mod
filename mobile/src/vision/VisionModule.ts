import { NativeEventEmitter, NativeModules } from "react-native";
import type { HazardWarning } from "../types";

type VisionNativeModule = {
  start(config: Record<string, unknown>): Promise<void>;
  stop(): Promise<void>;
};

const native = NativeModules.LAHacksVision as VisionNativeModule | undefined;

export const hasNativeVision = Boolean(native);

export async function startVision(config: Record<string, unknown>): Promise<void> {
  if (!native) {
    return;
  }
  await native.start(config);
}

export async function stopVision(): Promise<void> {
  if (!native) {
    return;
  }
  await native.stop();
}

export function subscribeWarnings(onWarning: (warning: HazardWarning) => void): () => void {
  if (!native) {
    return () => undefined;
  }
  const emitter = new NativeEventEmitter(native as never);
  const sub = emitter.addListener("visionWarning", (payload: unknown) => {
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
  return () => sub.remove();
}
