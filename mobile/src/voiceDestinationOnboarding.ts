import * as Speech from "expo-speech";
import { requireOptionalNativeModule } from "expo-modules-core";

/** Subset of `ExpoSpeechRecognitionModule` used here (avoids coupling to full native typings). */
type RecognitionModule = {
  requestPermissionsAsync: () => Promise<{ granted: boolean }>;
  start: (opts: Record<string, unknown>) => void;
  stop: () => void;
  abort: () => void;
  addListener: (event: string, cb: (ev: unknown) => void) => { remove: () => void };
};

/**
 * Prefer `requireOptionalNativeModule` over `require("expo-speech-recognition")`.
 * The npm package's entry file calls `requireNativeModule` and throws if the dev client on device
 * was built before `expo-speech-recognition` was added — optional lookup returns null instead.
 */
function loadRecognitionModule(): RecognitionModule | null {
  return requireOptionalNativeModule<RecognitionModule>("ExpoSpeechRecognition");
}

type SpeechResultEvent = {
  isFinal: boolean;
  results: { transcript: string }[];
};

type SpeechErrorEvent = {
  error?: string;
};

export function speakAsync(message: string, urgent = false): Promise<void> {
  return new Promise((resolve) => {
    Speech.speak(message, {
      rate: urgent ? 1.02 : 0.92,
      pitch: urgent ? 1.05 : 1.0,
      language: "en-US",
      onDone: () => resolve(),
      onStopped: () => resolve(),
      onError: () => resolve()
    });
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function isAffirmative(text: string): boolean {
  const t = text.toLowerCase().trim();
  if (t === "y") return true;
  return /\b(yes|yeah|yep|correct|right|sure|affirmative|uh-huh|uh huh)\b/.test(t);
}

function isNegative(text: string): boolean {
  const t = text.toLowerCase().trim();
  if (t === "n") return true;
  return /\b(no|nope|wrong|incorrect|negative|cancel)\b/.test(t);
}

async function listenOnce(module: RecognitionModule, maxMs: number): Promise<string | null> {
  await Speech.stop().catch(() => undefined);

  return new Promise((resolve) => {
    let settled = false;
    let lastTranscript = "";
    const subs: { remove: () => void }[] = [];
    let timer: ReturnType<typeof setTimeout> | undefined;

    const finish = (value: string | null): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (timer !== undefined) {
        clearTimeout(timer);
      }
      try {
        module.abort();
      } catch {
        try {
          module.stop();
        } catch {
          /* ignore */
        }
      }
      subs.forEach((s) => s.remove());
      resolve(value);
    };

    timer = setTimeout(() => finish(null), maxMs);

    subs.push(
      module.addListener("result", (ev: unknown) => {
        const e = ev as SpeechResultEvent;
        const t = e.results[0]?.transcript?.trim();
        if (t) {
          lastTranscript = t;
        }
        if (e.isFinal && lastTranscript) {
          finish(lastTranscript);
        }
      })
    );

    subs.push(
      module.addListener("error", (ev: unknown) => {
        const e = ev as SpeechErrorEvent;
        if (settled) {
          return;
        }
        if (e.error === "aborted") {
          return;
        }
        if (e.error === "no-speech" && lastTranscript) {
          finish(lastTranscript);
          return;
        }
        finish(null);
      })
    );

    subs.push(
      module.addListener("nomatch", () => {
        if (!settled) {
          finish(null);
        }
      })
    );

    subs.push(
      module.addListener("end", () => {
        if (settled) {
          return;
        }
        finish(lastTranscript.trim() ? lastTranscript.trim() : null);
      })
    );

    try {
      module.start({
        lang: "en-US",
        interimResults: true,
        continuous: false,
        maxAlternatives: 1,
        iosVoiceProcessingEnabled: true
      });
    } catch {
      finish(null);
    }
  });
}

/**
 * Prompts for a spoken destination, confirms it, and returns the confirmed string.
 * Returns null if speech is unavailable, permission denied, or the flow is abandoned.
 */
export async function runVoiceDestinationOnboarding(log: (line: string) => void): Promise<string | null> {
  const mod = loadRecognitionModule();
  if (!mod) {
    log("Voice destination skipped (speech recognition needs a dev build with native modules).");
    return null;
  }

  let perm: { granted: boolean };
  try {
    perm = await mod.requestPermissionsAsync();
  } catch {
    log("Voice destination skipped (speech recognition unavailable).");
    return null;
  }

  if (!perm.granted) {
    log("Voice destination skipped (microphone or speech recognition not allowed).");
    await speakAsync("You can type your destination on screen.");
    return null;
  }

  for (let attempt = 0; attempt < 3; attempt++) {
    await speakAsync("Where would you like to go?");
    await delay(450);
    const place = await listenOnce(mod, 22_000);
    if (!place) {
      await speakAsync("I did not catch that. Please try again.");
      continue;
    }

    await speakAsync(`Did you say ${place}?`);
    await delay(450);
    const answer = await listenOnce(mod, 16_000);
    if (answer == null) {
      await speakAsync("I did not hear yes or no. Let me ask again.");
      continue;
    }
    if (isAffirmative(answer)) {
      await speakAsync(`Okay. Your destination is set to ${place}. Load your route when you are ready.`);
      log(`Voice destination confirmed: ${place}`);
      return place;
    }
    if (isNegative(answer)) {
      await speakAsync("No problem. Let's try again.");
      continue;
    }

    await speakAsync("Please say yes if that is correct, or no to try again.");
    await delay(400);
    const clarify = await listenOnce(mod, 14_000);
    if (clarify && isAffirmative(clarify)) {
      await speakAsync(`Okay. Your destination is set to ${place}. Load your route when you are ready.`);
      log(`Voice destination confirmed: ${place}`);
      return place;
    }
    if (clarify && isNegative(clarify)) {
      await speakAsync("Let's try again.");
      continue;
    }
  }

  await speakAsync("You can type your destination on screen whenever you like.");
  log("Voice destination setup ended without a confirmed place.");
  return null;
}
