import Constants from "expo-constants";
import * as Location from "expo-location";
import * as Speech from "expo-speech";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View
} from "react-native";

import { fetchGoogleWalkingSteps } from "./src/maps/googleDirections";
import { haversineMeters, progressNavigation } from "./src/navigationEngine";
import type { HazardWarning, LatLng, RouteStep } from "./src/types";
import { runVoiceDestinationOnboarding } from "./src/voiceDestinationOnboarding";
import { hasNativeVision, startVision, stopVision, subscribeWarnings } from "./src/vision/VisionModule";

const ARRIVAL_RADIUS_M = 14;
const CONFIRM_HITS_NEEDED = 2;
const ROUTE_REPEAT_MS = 25_000;
const WARNING_PAUSE_MS = 4_500;

type Phase = "idle" | "locating" | "routing" | "navigating" | "arrived" | "error";
type VisionState = "unavailable" | "off" | "starting" | "on" | "error";

function getGoogleMapsApiKey(): string {
  const extra = Constants.expoConfig?.extra as { googleMapsApiKey?: string } | undefined;
  return extra?.googleMapsApiKey?.trim() ?? "";
}

function toLatLng(location: Location.LocationObject): LatLng {
  return {
    lat: location.coords.latitude,
    lon: location.coords.longitude
  };
}

function feet(meters: number | null): string {
  if (meters == null || !Number.isFinite(meters)) {
    return "--";
  }
  return `${Math.max(0, Math.round(meters * 3.28084))} ft`;
}

function formatTime(): string {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
}

export default function App(): React.JSX.Element {
  const googleMapsApiKey = useMemo(getGoogleMapsApiKey, []);

  const [destination, setDestination] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [visionState, setVisionState] = useState<VisionState>(hasNativeVision ? "off" : "unavailable");
  const [currentLocation, setCurrentLocation] = useState<LatLng | null>(null);
  const [routeSteps, setRouteSteps] = useState<RouteStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [distanceToTargetM, setDistanceToTargetM] = useState<number | null>(null);
  const [lastWarning, setLastWarning] = useState<HazardWarning | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);

  const watchRef = useRef<Location.LocationSubscription | null>(null);
  const stepsRef = useRef<RouteStep[]>([]);
  const currentIndexRef = useRef(0);
  const confirmedHitsRef = useRef(0);
  const navigatingRef = useRef(false);
  const lastRouteSpeakAtRef = useRef(0);
  const lastWarningAtRef = useRef(0);

  const currentStep = routeSteps[currentStepIndex] ?? null;
  const destinationText = destination.trim();
  const isBusy = phase === "locating" || phase === "routing";
  const isNavigating = phase === "navigating";

  const addLog = useCallback((line: string) => {
    setLogLines((prev) => [`${formatTime()} ${line}`, ...prev].slice(0, 8));
  }, []);

  const speak = useCallback(async (message: string, urgent = false): Promise<void> => {
    if (!message.trim()) {
      return;
    }
    if (urgent) {
      lastWarningAtRef.current = Date.now();
      await Speech.stop().catch(() => undefined);
    }
    await new Promise<void>((resolve) => {
      Speech.speak(message, {
        language: "en-US",
        rate: urgent ? 1.02 : 0.92,
        pitch: urgent ? 1.05 : 1.0,
        onDone: () => resolve(),
        onStopped: () => resolve(),
        onError: () => resolve()
      });
    });
  }, []);

  const removeLocationWatch = useCallback(() => {
    if (watchRef.current) {
      watchRef.current.remove();
      watchRef.current = null;
    }
  }, []);

  const resetRouteState = useCallback(() => {
    navigatingRef.current = false;
    stepsRef.current = [];
    currentIndexRef.current = 0;
    confirmedHitsRef.current = 0;
    lastRouteSpeakAtRef.current = 0;
    setRouteSteps([]);
    setCurrentStepIndex(0);
    setDistanceToTargetM(null);
  }, []);

  const handleLocationUpdate = useCallback(
    (location: Location.LocationObject) => {
      const nextLocation = toLatLng(location);
      setCurrentLocation(nextLocation);

      const steps = stepsRef.current;
      if (!navigatingRef.current || steps.length === 0) {
        return;
      }

      const index = currentIndexRef.current;
      if (index >= steps.length) {
        return;
      }

      const result = progressNavigation(
        steps,
        index,
        nextLocation,
        ARRIVAL_RADIUS_M,
        confirmedHitsRef.current,
        CONFIRM_HITS_NEEDED
      );

      confirmedHitsRef.current = result.confirmedHits;
      setDistanceToTargetM(result.distanceToTargetM);

      if (result.reachedDestination) {
        navigatingRef.current = false;
        currentIndexRef.current = steps.length;
        setCurrentStepIndex(steps.length);
        setPhase("arrived");
        removeLocationWatch();
        addLog("Destination area reached");
        void speak("You have reached the destination area. Obstacle warnings are still active.");
        return;
      }

      if (result.reachedStep) {
        const nextIndex = result.nextIndex;
        currentIndexRef.current = nextIndex;
        setCurrentStepIndex(nextIndex);
        confirmedHitsRef.current = 0;
        const instruction = steps[nextIndex]?.instruction;
        if (instruction) {
          lastRouteSpeakAtRef.current = Date.now();
          addLog(`Next step: ${instruction}`);
          void speak(`Next: ${instruction}`);
        }
        return;
      }

      const now = Date.now();
      const shouldRepeatRoute =
        now - lastRouteSpeakAtRef.current >= ROUTE_REPEAT_MS && now - lastWarningAtRef.current >= WARNING_PAUSE_MS;
      if (shouldRepeatRoute) {
        const instruction = steps[index]?.instruction;
        if (instruction) {
          lastRouteSpeakAtRef.current = now;
          void speak(`${instruction}. In about ${feet(result.distanceToTargetM)}.`);
        }
      }
    },
    [addLog, removeLocationWatch, speak]
  );

  const startLocationWatch = useCallback(async () => {
    removeLocationWatch();
    watchRef.current = await Location.watchPositionAsync(
      {
        accuracy: Location.Accuracy.BestForNavigation,
        distanceInterval: 1,
        timeInterval: 1000
      },
      handleLocationUpdate
    );
  }, [handleLocationUpdate, removeLocationWatch]);

  const ensureLocationPermission = useCallback(async (): Promise<boolean> => {
    const existing = await Location.getForegroundPermissionsAsync();
    if (existing.granted) {
      return true;
    }
    const requested = await Location.requestForegroundPermissionsAsync();
    return requested.granted;
  }, []);

  const startVisionIfAvailable = useCallback(async () => {
    if (!hasNativeVision) {
      setVisionState("unavailable");
      addLog("Native vision module unavailable in this build");
      return;
    }
    try {
      setVisionState("starting");
      await startVision({ confirmFrames: 2, warningCooldownS: 2.5 });
      setVisionState("on");
      addLog("Obstacle detection started");
    } catch (error) {
      setVisionState("error");
      addLog(`Vision failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [addLog]);

  const loadRoute = useCallback(async () => {
    if (isBusy) {
      return;
    }
    if (!destinationText) {
      setPhase("error");
      addLog("Destination is empty");
      void speak("Enter a destination first.");
      return;
    }
    if (!googleMapsApiKey) {
      setPhase("error");
      addLog("Missing Google Maps API key in app config");
      void speak("Google Maps API key is missing.");
      return;
    }

    try {
      setPhase("locating");
      addLog("Requesting current location");
      const allowed = await ensureLocationPermission();
      if (!allowed) {
        setPhase("error");
        addLog("Location permission denied");
        void speak("Location permission is needed for navigation.");
        return;
      }

      const originFix = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.BestForNavigation
      });
      const origin = toLatLng(originFix);
      setCurrentLocation(origin);

      setPhase("routing");
      addLog(`Loading walking route to ${destinationText}`);
      const steps = await fetchGoogleWalkingSteps(googleMapsApiKey, origin, destinationText);

      stepsRef.current = steps;
      currentIndexRef.current = 0;
      confirmedHitsRef.current = 0;
      navigatingRef.current = true;
      lastRouteSpeakAtRef.current = Date.now();
      setRouteSteps(steps);
      setCurrentStepIndex(0);
      setDistanceToTargetM(haversineMeters(origin, steps[0].end));
      setPhase("navigating");

      await startLocationWatch();
      await startVisionIfAvailable();

      addLog(`Route loaded with ${steps.length} steps`);
      void speak(`Route loaded. Step 1: ${steps[0].instruction}`);
    } catch (error) {
      navigatingRef.current = false;
      setPhase("error");
      addLog(`Route failed: ${error instanceof Error ? error.message : String(error)}`);
      void speak("I could not load that route. Check the destination and network connection.");
    }
  }, [
    addLog,
    destinationText,
    ensureLocationPermission,
    googleMapsApiKey,
    isBusy,
    speak,
    startLocationWatch,
    startVisionIfAvailable
  ]);

  const stopAll = useCallback(async () => {
    removeLocationWatch();
    resetRouteState();
    setPhase("idle");
    try {
      await stopVision();
      setVisionState(hasNativeVision ? "off" : "unavailable");
    } catch (error) {
      setVisionState("error");
      addLog(`Vision stop failed: ${error instanceof Error ? error.message : String(error)}`);
    }
    await Speech.stop().catch(() => undefined);
    addLog("Navigation stopped");
  }, [addLog, removeLocationWatch, resetRouteState]);

  const useVoiceDestination = useCallback(async () => {
    const heard = await runVoiceDestinationOnboarding(addLog);
    if (heard) {
      setDestination(heard);
    }
  }, [addLog]);

  useEffect(() => {
    const unsubscribe = subscribeWarnings((warning) => {
      setLastWarning(warning);
      addLog(warning.message);
      void speak(warning.message, warning.level === "urgent");
    });
    return () => {
      unsubscribe();
      removeLocationWatch();
      void stopVision();
      void Speech.stop();
    };
  }, [addLog, removeLocationWatch, speak]);

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.kicker}>LAHacks Nav</Text>
        <Text style={styles.title}>Walking Route</Text>
        <View style={styles.statusRow}>
          <StatusPill label={phase} tone={phase === "error" ? "danger" : isNavigating ? "active" : "neutral"} />
          <StatusPill label={`vision ${visionState}`} tone={visionState === "on" ? "active" : visionState === "error" ? "danger" : "neutral"} />
        </View>
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentInner} keyboardShouldPersistTaps="handled">
        <View style={styles.panel}>
          <Text style={styles.label}>Destination</Text>
          <TextInput
            value={destination}
            onChangeText={setDestination}
            placeholder="Enter an address or place"
            placeholderTextColor="#6b7280"
            style={styles.input}
            autoCapitalize="words"
            returnKeyType="go"
            onSubmitEditing={() => {
              void loadRoute();
            }}
          />
          <View style={styles.buttonRow}>
            <ActionButton label="Voice" onPress={useVoiceDestination} disabled={isBusy || isNavigating} variant="secondary" />
            <ActionButton label={isBusy ? "Loading" : "Start"} onPress={loadRoute} disabled={isBusy || !destinationText} />
            <ActionButton label="Stop" onPress={stopAll} disabled={phase === "idle" && visionState !== "on"} variant="danger" />
          </View>
        </View>

        <View style={styles.routePanel}>
          <View style={styles.routeHeader}>
            <Text style={styles.sectionTitle}>Current Step</Text>
            {isBusy ? <ActivityIndicator color="#0f766e" /> : null}
          </View>
          <Text style={styles.stepCount}>
            {routeSteps.length > 0 ? `${Math.min(currentStepIndex + 1, routeSteps.length)} of ${routeSteps.length}` : "No route loaded"}
          </Text>
          <Text style={styles.currentInstruction}>
            {currentStep?.instruction ?? (phase === "arrived" ? "Destination area reached." : "Set a destination and start navigation.")}
          </Text>
          <Text style={styles.distanceText}>Next target: {feet(distanceToTargetM)}</Text>
        </View>

        {lastWarning ? (
          <View style={[styles.warningPanel, lastWarning.level === "urgent" ? styles.warningUrgent : styles.warningNormal]}>
            <Text style={styles.warningLabel}>{lastWarning.level === "urgent" ? "Urgent Warning" : "Warning"}</Text>
            <Text style={styles.warningMessage}>{lastWarning.message}</Text>
          </View>
        ) : null}

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>Location</Text>
          <Text style={styles.metaText}>
            {currentLocation ? `${currentLocation.lat.toFixed(6)}, ${currentLocation.lon.toFixed(6)}` : "Waiting for GPS"}
          </Text>
        </View>

        {routeSteps.length > 0 ? (
          <View style={styles.panel}>
            <Text style={styles.sectionTitle}>Route Steps</Text>
            {routeSteps.map((step, index) => (
              <View key={`${index}-${step.instruction}`} style={[styles.stepRow, index === currentStepIndex ? styles.stepRowActive : null]}>
                <Text style={[styles.stepNumber, index === currentStepIndex ? styles.stepNumberActive : null]}>{index + 1}</Text>
                <Text style={[styles.stepText, index === currentStepIndex ? styles.stepTextActive : null]}>{step.instruction}</Text>
              </View>
            ))}
          </View>
        ) : null}

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>Activity</Text>
          {logLines.length === 0 ? <Text style={styles.metaText}>No activity yet.</Text> : null}
          {logLines.map((line) => (
            <Text key={line} style={styles.logText}>
              {line}
            </Text>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function StatusPill({ label, tone }: { label: string; tone: "active" | "danger" | "neutral" }): React.JSX.Element {
  return (
    <View style={[styles.statusPill, tone === "active" ? styles.statusActive : tone === "danger" ? styles.statusDanger : null]}>
      <Text style={[styles.statusText, tone === "active" ? styles.statusTextActive : tone === "danger" ? styles.statusTextDanger : null]}>
        {label}
      </Text>
    </View>
  );
}

function ActionButton({
  label,
  onPress,
  disabled,
  variant = "primary"
}: {
  label: string;
  onPress: () => void | Promise<void>;
  disabled?: boolean;
  variant?: "primary" | "secondary" | "danger";
}): React.JSX.Element {
  return (
    <Pressable
      accessibilityRole="button"
      disabled={disabled}
      onPress={() => {
        void onPress();
      }}
      style={({ pressed }) => [
        styles.button,
        variant === "secondary" ? styles.buttonSecondary : null,
        variant === "danger" ? styles.buttonDanger : null,
        pressed && !disabled ? styles.buttonPressed : null,
        disabled ? styles.buttonDisabled : null
      ]}
    >
      <Text
        style={[
          styles.buttonText,
          variant === "secondary" ? styles.buttonTextSecondary : null,
          variant === "danger" ? styles.buttonTextDanger : null,
          disabled ? styles.buttonTextDisabled : null
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#eef2f3"
  },
  header: {
    backgroundColor: "#102a2d",
    paddingHorizontal: 20,
    paddingBottom: 18,
    paddingTop: 18
  },
  kicker: {
    color: "#9dd3c7",
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0,
    textTransform: "uppercase"
  },
  title: {
    color: "#ffffff",
    fontSize: 30,
    fontWeight: "800",
    letterSpacing: 0,
    marginTop: 4
  },
  statusRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 14
  },
  statusPill: {
    borderColor: "#6b7c80",
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 5
  },
  statusActive: {
    backgroundColor: "#d7f3e7",
    borderColor: "#54b58e"
  },
  statusDanger: {
    backgroundColor: "#ffe4df",
    borderColor: "#f9735b"
  },
  statusText: {
    color: "#d4dee0",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0,
    textTransform: "uppercase"
  },
  statusTextActive: {
    color: "#0f5138"
  },
  statusTextDanger: {
    color: "#9f1d14"
  },
  content: {
    flex: 1
  },
  contentInner: {
    gap: 12,
    padding: 14,
    paddingBottom: 36
  },
  panel: {
    backgroundColor: "#ffffff",
    borderColor: "#d8dee2",
    borderRadius: 8,
    borderWidth: 1,
    padding: 14
  },
  routePanel: {
    backgroundColor: "#ffffff",
    borderColor: "#b7d4d0",
    borderRadius: 8,
    borderWidth: 1,
    padding: 16
  },
  label: {
    color: "#334155",
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 8
  },
  input: {
    backgroundColor: "#f8fafc",
    borderColor: "#b9c4cc",
    borderRadius: 6,
    borderWidth: 1,
    color: "#111827",
    fontSize: 17,
    minHeight: 48,
    paddingHorizontal: 12,
    paddingVertical: 10
  },
  buttonRow: {
    flexDirection: "row",
    gap: 8,
    marginTop: 12
  },
  button: {
    alignItems: "center",
    backgroundColor: "#0f766e",
    borderColor: "#0f766e",
    borderRadius: 6,
    borderWidth: 1,
    flex: 1,
    minHeight: 46,
    justifyContent: "center",
    paddingHorizontal: 10
  },
  buttonSecondary: {
    backgroundColor: "#ffffff",
    borderColor: "#64748b"
  },
  buttonDanger: {
    backgroundColor: "#fff7ed",
    borderColor: "#ea580c"
  },
  buttonPressed: {
    opacity: 0.75
  },
  buttonDisabled: {
    backgroundColor: "#e5e7eb",
    borderColor: "#d1d5db"
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 15,
    fontWeight: "800"
  },
  buttonTextSecondary: {
    color: "#334155"
  },
  buttonTextDanger: {
    color: "#9a3412"
  },
  buttonTextDisabled: {
    color: "#6b7280"
  },
  routeHeader: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between"
  },
  sectionTitle: {
    color: "#111827",
    fontSize: 16,
    fontWeight: "800",
    letterSpacing: 0
  },
  stepCount: {
    color: "#64748b",
    fontSize: 13,
    fontWeight: "700",
    marginTop: 8
  },
  currentInstruction: {
    color: "#0f172a",
    fontSize: 22,
    fontWeight: "800",
    letterSpacing: 0,
    lineHeight: 29,
    marginTop: 8
  },
  distanceText: {
    color: "#0f766e",
    fontSize: 15,
    fontWeight: "800",
    marginTop: 10
  },
  warningPanel: {
    borderRadius: 8,
    borderWidth: 1,
    padding: 14
  },
  warningUrgent: {
    backgroundColor: "#fff1f2",
    borderColor: "#fb7185"
  },
  warningNormal: {
    backgroundColor: "#fffbeb",
    borderColor: "#f59e0b"
  },
  warningLabel: {
    color: "#9f1239",
    fontSize: 13,
    fontWeight: "900",
    textTransform: "uppercase"
  },
  warningMessage: {
    color: "#111827",
    fontSize: 20,
    fontWeight: "800",
    lineHeight: 27,
    marginTop: 5
  },
  metaText: {
    color: "#475569",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8
  },
  stepRow: {
    alignItems: "flex-start",
    borderTopColor: "#e5e7eb",
    borderTopWidth: 1,
    flexDirection: "row",
    gap: 10,
    paddingVertical: 10
  },
  stepRowActive: {
    backgroundColor: "#ecfdf5"
  },
  stepNumber: {
    color: "#64748b",
    fontSize: 13,
    fontWeight: "900",
    minWidth: 26,
    textAlign: "center"
  },
  stepNumberActive: {
    color: "#047857"
  },
  stepText: {
    color: "#334155",
    flex: 1,
    fontSize: 15,
    lineHeight: 21
  },
  stepTextActive: {
    color: "#064e3b",
    fontWeight: "800"
  },
  logText: {
    borderTopColor: "#edf2f7",
    borderTopWidth: 1,
    color: "#475569",
    fontSize: 13,
    lineHeight: 19,
    paddingVertical: 7
  }
});
