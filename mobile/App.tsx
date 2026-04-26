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

import { formatNavApiStatus, getNavApiBaseUrl } from "./src/cloudflareConfig";
import { fetchGoogleWalkingSteps } from "./src/maps/googleDirections";
import { haversineMeters, progressNavigation } from "./src/navigationEngine";
import type { HazardWarning, LatLng, RouteStep } from "./src/types";
import { runVoiceDestinationOnboarding } from "./src/voiceDestinationOnboarding";
import { hasNativeVision, startVision, stopVision, subscribeWarnings } from "./src/vision/VisionModule";

const ARRIVAL_RADIUS_M = 14;
const CONFIRM_HITS_NEEDED = 2;
const ROUTE_REPEAT_MS = 25_000;
const WARNING_PAUSE_MS = 4_500;
const WALKING_STEP_M = 0.75;

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

function walkingSteps(meters: number | null): string {
  if (meters == null || !Number.isFinite(meters)) {
    return "--";
  }
  const steps = Math.max(1, Math.round(meters / WALKING_STEP_M));
  return `${steps} ${steps === 1 ? "step" : "steps"}`;
}

function formatTime(): string {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
}

export default function App(): React.JSX.Element {
  const googleMapsApiKey = useMemo(getGoogleMapsApiKey, []);
  const navApiBaseUrl = useMemo(getNavApiBaseUrl, []);
  const navApiStatus = useMemo(() => formatNavApiStatus(navApiBaseUrl), [navApiBaseUrl]);

  const [destination, setDestination] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [visionState, setVisionState] = useState<VisionState>(hasNativeVision ? "off" : "unavailable");
  const [currentLocation, setCurrentLocation] = useState<LatLng | null>(null);
  const [routeSteps, setRouteSteps] = useState<RouteStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [distanceToTargetM, setDistanceToTargetM] = useState<number | null>(null);
  const [lastWarning, setLastWarning] = useState<HazardWarning | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [clockText, setClockText] = useState(formatTime());

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
  const isVisionActive = visionState === "on";
  const visionCopy =
    visionState === "on"
      ? "AI Vision Active"
      : visionState === "starting"
        ? "AI Vision Starting"
        : visionState === "error"
          ? "AI Vision Error"
          : hasNativeVision
            ? "AI Vision Ready"
            : "AI Vision Offline";
  const routeCopy = currentStep?.instruction ?? (phase === "arrived" ? "Destination area reached" : "Set destination and start");
  const warningCopy = lastWarning?.message ?? (isVisionActive ? "No obstacle warning" : "Vision standby");
  const secondaryAlertCopy =
    phase === "error"
      ? "Check setup and try again"
      : distanceToTargetM != null
        ? `Next target in ${walkingSteps(distanceToTargetM)}`
        : navApiBaseUrl
          ? "Cloud detector linked"
          : "Local detector mode";

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
          void speak(`${instruction}. In about ${walkingSteps(result.distanceToTargetM)}.`);
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
      await startVision({ confirmFrames: 2, warningCooldownS: 2.5, navApiBaseUrl: navApiBaseUrl || undefined });
      setVisionState("on");
      addLog(navApiBaseUrl ? `Obstacle detection started; cloud API ${navApiBaseUrl}` : "Obstacle detection started");
    } catch (error) {
      setVisionState("error");
      addLog(`Vision failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [addLog, navApiBaseUrl]);

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

  useEffect(() => {
    const timer = setInterval(() => {
      setClockText(formatTime());
    }, 15_000);
    return () => {
      clearInterval(timer);
    };
  }, []);

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.topBar}>
        <Text style={[styles.topStatus, isVisionActive ? styles.topStatusActive : styles.topStatusIdle]} numberOfLines={1}>
          {visionCopy}
        </Text>
        <Text style={styles.topClock}>{clockText}</Text>
        <Text style={styles.topMode}>{navApiBaseUrl ? "Cloud" : "Local"}</Text>
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentInner} keyboardShouldPersistTaps="handled">
        <View style={styles.hero}>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Set destination by voice"
            disabled={isBusy || isNavigating}
            onPress={() => {
              void useVoiceDestination();
            }}
            style={({ pressed }) => [styles.micButton, pressed && !isBusy && !isNavigating ? styles.micButtonPressed : null]}
          >
            <MicGlyph />
          </Pressable>
          <VoiceBars active={isVisionActive || isBusy} />
        </View>

        <View style={styles.destinationPill}>
          <PinGlyph />
          <TextInput
            value={destination}
            onChangeText={setDestination}
            placeholder="Main Entrance"
            placeholderTextColor="#ffffff"
            style={styles.destinationInput}
            autoCapitalize="words"
            returnKeyType="go"
            onSubmitEditing={() => {
              void loadRoute();
            }}
          />
        </View>

        <Pressable
          accessibilityRole="button"
          disabled={isBusy || !destinationText}
          onPress={() => {
            void loadRoute();
          }}
          style={({ pressed }) => [styles.guidanceCard, pressed && !isBusy && destinationText ? styles.cardPressed : null]}
        >
          <NavGlyph />
          <View style={styles.guidanceTextWrap}>
            <Text style={styles.guidanceText} numberOfLines={2}>
              {routeCopy}
            </Text>
            {routeSteps.length > 0 ? (
              <Text style={styles.guidanceMeta}>
                {Math.min(currentStepIndex + 1, routeSteps.length)} of {routeSteps.length} steps
              </Text>
            ) : null}
          </View>
          {isBusy ? <ActivityIndicator color="#3b82f6" /> : null}
        </Pressable>

        <View style={[styles.alertCard, lastWarning?.level === "urgent" ? styles.alertUrgent : styles.alertNormal]}>
          <AlertGlyph tone={lastWarning?.level === "urgent" ? "urgent" : "normal"} />
          <Text style={[styles.alertText, !lastWarning ? styles.alertTextIdle : null]} numberOfLines={2}>
            {warningCopy}
          </Text>
        </View>

        <View style={[styles.alertCard, phase === "error" ? styles.alertDanger : styles.alertMuted]}>
          <AlertGlyph tone={phase === "error" ? "danger" : "muted"} />
          <Text style={[styles.alertText, phase === "error" ? null : styles.alertTextMuted]} numberOfLines={2}>
            {secondaryAlertCopy}
          </Text>
        </View>

        <View style={styles.buttonRow}>
          <ActionButton label="Voice" onPress={useVoiceDestination} disabled={isBusy || isNavigating} variant="secondary" />
          <ActionButton label={isBusy ? "Loading" : isNavigating ? "Reroute" : "Start"} onPress={loadRoute} disabled={isBusy || !destinationText} />
          <ActionButton label="Stop" onPress={stopAll} disabled={phase === "idle" && visionState !== "on"} variant="danger" />
        </View>

        <View style={styles.statusShelf}>
          <StatusPill label={phase} tone={phase === "error" ? "danger" : isNavigating ? "active" : "neutral"} />
          <StatusPill label={`vision ${visionState}`} tone={visionState === "on" ? "active" : visionState === "error" ? "danger" : "neutral"} />
          <StatusPill label={navApiStatus} tone={navApiBaseUrl ? "active" : "neutral"} />
        </View>

        {routeSteps.length > 0 ? (
          <View style={styles.detailPanel}>
            {routeSteps.map((step, index) => (
              <View key={`${index}-${step.instruction}`} style={[styles.stepRow, index === currentStepIndex ? styles.stepRowActive : null]}>
                <Text style={[styles.stepNumber, index === currentStepIndex ? styles.stepNumberActive : null]}>{index + 1}</Text>
                <Text style={[styles.stepText, index === currentStepIndex ? styles.stepTextActive : null]}>{step.instruction}</Text>
              </View>
            ))}
          </View>
        ) : null}

        <View style={styles.detailPanel}>
          <Text style={styles.detailTitle}>
            {currentLocation ? `${currentLocation.lat.toFixed(6)}, ${currentLocation.lon.toFixed(6)}` : "Waiting for GPS"}
          </Text>
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

function MicGlyph(): React.JSX.Element {
  return (
    <View style={styles.micGlyph}>
      <View style={styles.micHead} />
      <View style={styles.micArc} />
      <View style={styles.micStem} />
      <View style={styles.micBase} />
    </View>
  );
}

function VoiceBars({ active }: { active: boolean }): React.JSX.Element {
  return (
    <View style={styles.voiceBars} accessibilityElementsHidden>
      {[0, 1, 2, 3, 4].map((bar) => (
        <View
          key={bar}
          style={[
            styles.voiceBar,
            active ? styles.voiceBarActive : styles.voiceBarIdle,
            bar === 0 || bar === 4 ? styles.voiceBarShort : null,
            bar === 1 || bar === 3 ? styles.voiceBarMedium : null,
            bar === 2 ? styles.voiceBarTall : null
          ]}
        />
      ))}
    </View>
  );
}

function PinGlyph(): React.JSX.Element {
  return (
    <View style={styles.pinGlyph}>
      <View style={styles.pinDot} />
    </View>
  );
}

function NavGlyph(): React.JSX.Element {
  return (
    <View style={styles.navGlyph}>
      <View style={styles.navNeedle} />
    </View>
  );
}

function AlertGlyph({ tone }: { tone: "normal" | "urgent" | "danger" | "muted" }): React.JSX.Element {
  return (
    <View style={[styles.alertGlyph, tone === "danger" ? styles.alertGlyphDanger : tone === "muted" ? styles.alertGlyphMuted : null]}>
      <Text style={[styles.alertGlyphText, tone === "danger" ? styles.alertGlyphTextDanger : tone === "muted" ? styles.alertGlyphTextMuted : null]}>
        !
      </Text>
    </View>
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
    backgroundColor: "#030712"
  },
  topBar: {
    alignItems: "center",
    backgroundColor: "#050814",
    borderBottomColor: "#111827",
    borderBottomWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 58,
    paddingHorizontal: 28
  },
  topStatus: {
    flex: 1,
    fontSize: 15,
    fontWeight: "800",
    letterSpacing: 0
  },
  topStatusActive: {
    color: "#22c55e"
  },
  topStatusIdle: {
    color: "#93a4b8"
  },
  topClock: {
    color: "#f8fafc",
    flex: 1,
    fontSize: 16,
    fontWeight: "800",
    textAlign: "center"
  },
  topMode: {
    color: "#f8fafc",
    flex: 1,
    fontSize: 15,
    fontWeight: "800",
    textAlign: "right"
  },
  content: {
    flex: 1
  },
  contentInner: {
    alignItems: "center",
    gap: 16,
    minHeight: "100%",
    paddingHorizontal: 28,
    paddingBottom: 32,
    paddingTop: 56
  },
  hero: {
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 10,
    minHeight: 230
  },
  micButton: {
    alignItems: "center",
    backgroundColor: "#2f80f6",
    borderRadius: 96,
    height: 132,
    justifyContent: "center",
    shadowColor: "#2563eb",
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.22,
    shadowRadius: 26,
    width: 132
  },
  micButtonPressed: {
    opacity: 0.78,
    transform: [{ scale: 0.98 }]
  },
  micGlyph: {
    alignItems: "center",
    height: 76,
    justifyContent: "center",
    width: 58
  },
  micHead: {
    backgroundColor: "#ffffff",
    borderRadius: 14,
    height: 43,
    width: 24
  },
  micArc: {
    borderBottomColor: "#ffffff",
    borderBottomWidth: 6,
    borderLeftColor: "#ffffff",
    borderLeftWidth: 6,
    borderRadius: 22,
    borderRightColor: "#ffffff",
    borderRightWidth: 6,
    height: 34,
    marginTop: -24,
    width: 44
  },
  micStem: {
    backgroundColor: "#ffffff",
    height: 15,
    marginTop: -2,
    width: 6
  },
  micBase: {
    backgroundColor: "#ffffff",
    height: 5,
    width: 18
  },
  voiceBars: {
    alignItems: "flex-end",
    flexDirection: "row",
    gap: 8,
    height: 64,
    justifyContent: "center",
    marginTop: 30
  },
  voiceBar: {
    borderRadius: 8,
    width: 12
  },
  voiceBarActive: {
    backgroundColor: "#2f80f6"
  },
  voiceBarIdle: {
    backgroundColor: "#1e3a8a"
  },
  voiceBarShort: {
    height: 24
  },
  voiceBarMedium: {
    height: 42
  },
  voiceBarTall: {
    height: 54
  },
  destinationPill: {
    alignItems: "center",
    alignSelf: "center",
    backgroundColor: "#111827",
    borderColor: "#334863",
    borderRadius: 999,
    borderWidth: 2,
    flexDirection: "row",
    gap: 14,
    maxWidth: 390,
    minHeight: 76,
    paddingHorizontal: 28,
    width: "82%"
  },
  destinationInput: {
    color: "#ffffff",
    flex: 1,
    fontSize: 24,
    fontWeight: "800",
    minHeight: 52,
    paddingVertical: 6
  },
  pinGlyph: {
    alignItems: "center",
    borderColor: "#3b82f6",
    borderRadius: 16,
    borderWidth: 3,
    height: 30,
    justifyContent: "center",
    transform: [{ rotate: "45deg" }],
    width: 30
  },
  pinDot: {
    backgroundColor: "#3b82f6",
    borderRadius: 5,
    height: 9,
    width: 9
  },
  guidanceCard: {
    alignItems: "center",
    alignSelf: "stretch",
    backgroundColor: "#071d43",
    borderColor: "#1d6ff2",
    borderRadius: 24,
    borderWidth: 2,
    flexDirection: "row",
    gap: 18,
    minHeight: 88,
    paddingHorizontal: 28,
    paddingVertical: 18
  },
  cardPressed: {
    opacity: 0.78
  },
  navGlyph: {
    borderBottomColor: "transparent",
    borderBottomWidth: 14,
    borderLeftColor: "#4e9aff",
    borderLeftWidth: 22,
    borderTopColor: "transparent",
    borderTopWidth: 14,
    height: 0,
    transform: [{ rotate: "-38deg" }],
    width: 0
  },
  navNeedle: {
    backgroundColor: "#4e9aff",
    height: 24,
    left: -11,
    position: "absolute",
    top: -2,
    width: 4
  },
  guidanceTextWrap: {
    flex: 1
  },
  guidanceText: {
    color: "#ffffff",
    fontSize: 24,
    fontWeight: "800",
    letterSpacing: 0,
    lineHeight: 31
  },
  guidanceMeta: {
    color: "#82b4ff",
    fontSize: 13,
    fontWeight: "800",
    marginTop: 4
  },
  alertCard: {
    alignItems: "center",
    alignSelf: "stretch",
    borderRadius: 24,
    borderWidth: 2,
    flexDirection: "row",
    gap: 18,
    minHeight: 88,
    paddingHorizontal: 28,
    paddingVertical: 18
  },
  alertNormal: {
    backgroundColor: "#352500",
    borderColor: "#b78a00"
  },
  alertUrgent: {
    backgroundColor: "#3d2600",
    borderColor: "#facc15"
  },
  alertDanger: {
    backgroundColor: "#2a0509",
    borderColor: "#9f1239"
  },
  alertMuted: {
    backgroundColor: "#22080b",
    borderColor: "#7f1d1d"
  },
  alertGlyph: {
    alignItems: "center",
    borderColor: "#facc15",
    borderRadius: 18,
    borderWidth: 3,
    height: 36,
    justifyContent: "center",
    width: 36
  },
  alertGlyphDanger: {
    borderColor: "#f87171"
  },
  alertGlyphMuted: {
    borderColor: "#9ca3af",
    opacity: 0.65
  },
  alertGlyphText: {
    color: "#facc15",
    fontSize: 20,
    fontWeight: "900"
  },
  alertGlyphTextDanger: {
    color: "#f87171"
  },
  alertGlyphTextMuted: {
    color: "#9ca3af"
  },
  alertText: {
    color: "#ffffff",
    flex: 1,
    fontSize: 23,
    fontWeight: "800",
    letterSpacing: 0,
    lineHeight: 30
  },
  alertTextIdle: {
    color: "#f7d874"
  },
  alertTextMuted: {
    color: "#a49aa0"
  },
  buttonRow: {
    alignSelf: "stretch",
    flexDirection: "row",
    gap: 10,
    marginTop: 2
  },
  button: {
    alignItems: "center",
    backgroundColor: "#2563eb",
    borderColor: "#3b82f6",
    borderRadius: 18,
    borderWidth: 1,
    flex: 1,
    justifyContent: "center",
    minHeight: 54,
    paddingHorizontal: 10
  },
  buttonSecondary: {
    backgroundColor: "#101827",
    borderColor: "#31435f"
  },
  buttonDanger: {
    backgroundColor: "#2a0b10",
    borderColor: "#b91c1c"
  },
  buttonPressed: {
    opacity: 0.72
  },
  buttonDisabled: {
    backgroundColor: "#111827",
    borderColor: "#1f2937"
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 15,
    fontWeight: "900"
  },
  buttonTextSecondary: {
    color: "#dbeafe"
  },
  buttonTextDanger: {
    color: "#fecaca"
  },
  buttonTextDisabled: {
    color: "#64748b"
  },
  statusShelf: {
    alignSelf: "stretch",
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    justifyContent: "center"
  },
  statusPill: {
    backgroundColor: "#0f172a",
    borderColor: "#253448",
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 11,
    paddingVertical: 6
  },
  statusActive: {
    backgroundColor: "#052e1b",
    borderColor: "#16a34a"
  },
  statusDanger: {
    backgroundColor: "#3a0a0a",
    borderColor: "#dc2626"
  },
  statusText: {
    color: "#9ca3af",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 0,
    textTransform: "uppercase"
  },
  statusTextActive: {
    color: "#86efac"
  },
  statusTextDanger: {
    color: "#fecaca"
  },
  detailPanel: {
    alignSelf: "stretch",
    backgroundColor: "#080d18",
    borderColor: "#1f2937",
    borderRadius: 18,
    borderWidth: 1,
    padding: 14
  },
  detailTitle: {
    color: "#93a4b8",
    fontSize: 13,
    fontWeight: "800"
  },
  stepRow: {
    alignItems: "flex-start",
    borderTopColor: "#1f2937",
    borderTopWidth: 1,
    flexDirection: "row",
    gap: 10,
    paddingVertical: 10
  },
  stepRowActive: {
    backgroundColor: "#102545"
  },
  stepNumber: {
    color: "#64748b",
    fontSize: 13,
    fontWeight: "900",
    minWidth: 26,
    textAlign: "center"
  },
  stepNumberActive: {
    color: "#60a5fa"
  },
  stepText: {
    color: "#cbd5e1",
    flex: 1,
    fontSize: 15,
    lineHeight: 21
  },
  stepTextActive: {
    color: "#ffffff",
    fontWeight: "800"
  },
  logText: {
    borderTopColor: "#1f2937",
    borderTopWidth: 1,
    color: "#94a3b8",
    fontSize: 13,
    lineHeight: 19,
    paddingVertical: 7
  }
});
