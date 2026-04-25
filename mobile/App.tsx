import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import * as Speech from "expo-speech";
import * as Location from "expo-location";
import { progressNavigation } from "./src/navigationEngine";
import type { HazardWarning, LatLng, RouteStep } from "./src/types";
import { hasNativeVision, startVision, stopVision, subscribeWarnings } from "./src/vision/VisionModule";

const DEMO_STEPS: RouteStep[] = [
  { instruction: "Head north on Campus Drive", end: { lat: 37.428018, lon: -122.170082 } },
  { instruction: "Turn left onto Palm Drive", end: { lat: 37.430166, lon: -122.168997 } },
  { instruction: "Continue straight for 100 meters", end: { lat: 37.431339, lon: -122.16964 } }
];

function speak(message: string, urgent = false): void {
  Speech.speak(message, {
    rate: urgent ? 1.02 : 0.92,
    pitch: urgent ? 1.05 : 1.0,
    language: "en-US"
  });
}

export default function App(): React.JSX.Element {
  const [steps] = useState<RouteStep[]>(DEMO_STEPS);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [confirmHits, setConfirmHits] = useState(0);
  const [arrivalRadiusM, setArrivalRadiusM] = useState("14");
  const [confirmHitsNeeded, setConfirmHitsNeeded] = useState("2");
  const [latestLoc, setLatestLoc] = useState<LatLng | null>(null);
  const [distanceToStep, setDistanceToStep] = useState<number | null>(null);
  const [simInput, setSimInput] = useState("");
  const [logLines, setLogLines] = useState<string[]>([]);
  const [useGps, setUseGps] = useState(false);
  const watchRef = useRef<Location.LocationSubscription | null>(null);

  const done = currentIndex >= steps.length;
  const currentStep = done ? null : steps[currentIndex];

  const appendLog = (line: string): void => {
    setLogLines((prev) => [`${new Date().toLocaleTimeString()} ${line}`, ...prev].slice(0, 30));
  };

  const processLocation = (loc: LatLng): void => {
    setLatestLoc(loc);
    if (done) {
      setDistanceToStep(0);
      return;
    }
    const radius = Math.max(2, Number(arrivalRadiusM) || 14);
    const neededHits = Math.max(1, Number(confirmHitsNeeded) || 2);
    const result = progressNavigation(steps, currentIndex, loc, radius, confirmHits, neededHits);
    setDistanceToStep(result.distanceToTargetM);
    if (!result.reachedStep) {
      setConfirmHits(result.confirmedHits);
      return;
    }
    if (result.reachedDestination) {
      setCurrentIndex(result.nextIndex);
      setConfirmHits(0);
      speak("You have reached the destination area. Obstacle detection is still active.");
      appendLog("Reached destination area.");
      return;
    }
    setCurrentIndex(result.nextIndex);
    setConfirmHits(0);
    const next = steps[result.nextIndex];
    speak(`Next: ${next.instruction}`);
    appendLog(`Advanced to step ${result.nextIndex + 1}: ${next.instruction}`);
  };

  const submitSimulatedLocation = (): void => {
    const parts = simInput.split(",").map((x) => x.trim());
    if (parts.length !== 2) {
      appendLog('Invalid input. Use "lat,lon".');
      return;
    }
    const lat = Number(parts[0]);
    const lon = Number(parts[1]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      appendLog("Invalid coordinate values.");
      return;
    }
    setSimInput("");
    processLocation({ lat, lon });
  };

  useEffect(() => {
    const unsubscribe = subscribeWarnings((warning: HazardWarning) => {
      speak(warning.message, warning.level === "urgent");
      appendLog(`Vision warning: ${warning.message}`);
    });
    startVision({ model: "yolo11n", conf: 0.35, classes: ["person", "chair", "car", "door"] }).catch(
      () => undefined
    );
    return () => {
      unsubscribe();
      stopVision().catch(() => undefined);
    };
  }, []);

  useEffect(() => {
    if (!useGps) {
      watchRef.current?.remove();
      watchRef.current = null;
      return;
    }
    let mounted = true;
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (!mounted || status !== "granted") {
        appendLog("Location permission denied.");
        setUseGps(false);
        return;
      }
      watchRef.current = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.Balanced,
          distanceInterval: 3,
          timeInterval: 1500
        },
        (pos) => {
          processLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        }
      );
      appendLog("Using live GPS updates.");
    })().catch(() => {
      appendLog("Failed to start GPS watcher.");
      setUseGps(false);
    });
    return () => {
      mounted = false;
      watchRef.current?.remove();
      watchRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useGps, currentIndex, confirmHits, arrivalRadiusM, confirmHitsNeeded]);

  const statusText = useMemo(() => {
    if (!currentStep) {
      return "Destination reached";
    }
    const dist = distanceToStep == null ? "-" : `${Math.round(distanceToStep)} m`;
    return `Step ${currentIndex + 1}/${steps.length}: ${currentStep.instruction} (distance: ${dist})`;
  }, [currentStep, currentIndex, distanceToStep, steps.length]);

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>LAHacks iOS Refactor</Text>
        <Text style={styles.subtitle}>
          Expo app with live step progression and native vision module hook ({hasNativeVision ? "native ready" : "mock mode"}).
        </Text>

        <View style={styles.card}>
          <Text style={styles.label}>Navigation status</Text>
          <Text style={styles.value}>{statusText}</Text>
          <Text style={styles.small}>
            Latest location: {latestLoc ? `${latestLoc.lat.toFixed(6)}, ${latestLoc.lon.toFixed(6)}` : "none"}
          </Text>
        </View>

        <View style={styles.row}>
          <View style={styles.inputBlock}>
            <Text style={styles.label}>Arrival radius (m)</Text>
            <TextInput value={arrivalRadiusM} onChangeText={setArrivalRadiusM} style={styles.input} keyboardType="numeric" />
          </View>
          <View style={styles.inputBlock}>
            <Text style={styles.label}>Confirm hits</Text>
            <TextInput
              value={confirmHitsNeeded}
              onChangeText={setConfirmHitsNeeded}
              style={styles.input}
              keyboardType="numeric"
            />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Simulate location (lat,lon)</Text>
          <TextInput
            value={simInput}
            onChangeText={setSimInput}
            style={styles.input}
            placeholder="37.427500,-122.169700"
            autoCapitalize="none"
          />
          <View style={styles.rowButtons}>
            <Button title="Submit coordinate" onPress={submitSimulatedLocation} />
            <Button
              title={useGps ? "Stop GPS" : "Use GPS"}
              onPress={() => setUseGps((v) => !v)}
              color={useGps ? "#9b1c1c" : "#1d4ed8"}
            />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Recent events</Text>
          {logLines.length === 0 ? <Text style={styles.small}>No events yet.</Text> : null}
          {logLines.map((line) => (
            <Text key={line} style={styles.logLine}>
              {line}
            </Text>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#0b1220" },
  content: { padding: 16, gap: 14 },
  title: { color: "#f8fafc", fontSize: 26, fontWeight: "700" },
  subtitle: { color: "#cbd5e1", lineHeight: 20 },
  card: { backgroundColor: "#111827", borderRadius: 10, padding: 12, gap: 8 },
  label: { color: "#93c5fd", fontWeight: "600" },
  value: { color: "#f8fafc", fontSize: 16 },
  small: { color: "#94a3b8" },
  row: { flexDirection: "row", gap: 10 },
  inputBlock: { flex: 1, gap: 6 },
  input: { backgroundColor: "#1f2937", color: "#f8fafc", borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8 },
  rowButtons: { flexDirection: "row", justifyContent: "space-between", gap: 10 },
  logLine: { color: "#e2e8f0", fontSize: 12 }
});
