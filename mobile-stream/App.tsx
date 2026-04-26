import { CameraView, useCameraPermissions } from 'expo-camera';
import Constants from 'expo-constants';
import * as Network from 'expo-network';
import * as Speech from 'expo-speech';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

type StreamJson = {
  speak?: boolean;
  decision?: { message?: string; priority?: number; action?: string };
};

function normalizeBase(url: string): string {
  return url.trim().replace(/\/+$/, '');
}

const PING_TIMEOUT_MS = 45_000;
const SESSION_TIMEOUT_MS = 120_000;
const FRAME_TIMEOUT_MS = 180_000;

function isAbortError(e: unknown): boolean {
  if (e instanceof Error && e.name === 'AbortError') return true;
  const s = String(e).toLowerCase();
  return s.includes('abort') || s.includes('aborted');
}

/** Merge fetch init headers; ngrok free tier needs this for non-browser clients. */
function streamFetchInit(requestUrl: string, init: RequestInit = {}): RequestInit {
  const headers = new Headers((init.headers as HeadersInit | undefined) ?? undefined);
  let host = '';
  try {
    host = new URL(requestUrl).hostname.toLowerCase();
  } catch {
    /* ignore */
  }
  if (host.includes('ngrok')) {
    headers.set('ngrok-skip-browser-warning', '69420');
  }
  return { ...init, headers };
}

async function fetchWithTimeout(
  input: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(input, { ...streamFetchInit(input, init), signal: ctrl.signal });
  } finally {
    clearTimeout(t);
  }
}

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const camRef = useRef<CameraView>(null);

  const [baseUrl, setBaseUrl] = useState('http://192.168.1.1:8765');
  const [destination, setDestination] = useState('Library');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pinging, setPinging] = useState(false);
  const [log, setLog] = useState<string[]>([]);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startLockRef = useRef(false);
  const [streamStarting, setStreamStarting] = useState(false);
  const inFlightRef = useRef(false);
  const lastSpokenRef = useRef('');
  const lastSpokenAtRef = useRef(0);

  const appendLog = useCallback((line: string) => {
    setLog((prev) => {
      const next = [`${new Date().toLocaleTimeString()} ${line}`, ...prev];
      return next.slice(0, 40);
    });
  }, []);

  const maybeSpeak = useCallback((text: string, priority: number) => {
    if (!text) return;
    const now = Date.now();
    const urgent = priority >= 95;
    if (!urgent && text === lastSpokenRef.current && now - lastSpokenAtRef.current < 3500) {
      return;
    }
    lastSpokenRef.current = text;
    lastSpokenAtRef.current = now;
    Speech.stop();
    Speech.speak(text, { rate: 1.0 });
  }, []);

  const pingServer = useCallback(async () => {
    const base = normalizeBase(baseUrl);
    if (!base) {
      appendLog('Set server base URL first.');
      return;
    }
    setPinging(true);
    try {
      const res = await fetchWithTimeout(`${base}/ping`, { method: 'GET' }, PING_TIMEOUT_MS);
      const txt = await res.text();
      appendLog(res.ok ? `ping ok: ${txt.slice(0, 80)}` : `ping HTTP ${res.status}`);
    } catch (e) {
      appendLog(`ping failed: ${String(e)}`);
    } finally {
      setPinging(false);
    }
  }, [appendLog, baseUrl]);

  /** POST /stream/session; returns session_id or null. */
  const createSessionRequest = useCallback(async (): Promise<string | null> => {
    const base = normalizeBase(baseUrl);
    if (!base) {
      appendLog('Set server base URL first.');
      return null;
    }
    setBusy(true);
    try {
      try {
        const ns = await Network.getNetworkStateAsync();
        appendLog(
          `device network: type=${ns.type ?? '?'} connected=${String(ns.isConnected)} internet=${String(ns.isInternetReachable)}`,
        );
        if (ns.type === Network.NetworkStateType.CELLULAR) {
          appendLog(
            'On cellular data: http://192.168… cannot reach your laptop. Join the same Wi‑Fi as the computer, or use an https ngrok URL to stream_server.',
          );
        }
        if (
          Platform.OS === 'ios' &&
          Constants.appOwnership === 'expo' &&
          base.toLowerCase().startsWith('http://')
        ) {
          appendLog(
            'iOS Expo Go: http:// LAN URLs often time out. On Mac: brew install cloudflared && ./scripts/tunnel_stream_server.sh — paste the https trycloudflare URL here.',
          );
        }
      } catch {
        appendLog('device network: could not read state');
      }

      const fd = new FormData();
      fd.append('destination', destination.trim() || 'Unknown');
      const res = await fetchWithTimeout(`${base}/stream/session`, { method: 'POST', body: fd }, SESSION_TIMEOUT_MS);
      if (!res.ok) {
        appendLog(`session HTTP ${res.status}: ${await res.text()}`);
        setSessionId(null);
        return null;
      }
      const j = (await res.json()) as { session_id?: string };
      if (!j.session_id) {
        appendLog('session: missing session_id in JSON');
        setSessionId(null);
        return null;
      }
      setSessionId(j.session_id);
      appendLog(`session ok: ${j.session_id.slice(0, 8)}…`);
      return j.session_id;
    } catch (e) {
      if (isAbortError(e)) {
        appendLog(
          'Session request timed out (aborted). Often iOS is waiting: allow Local Network for Expo Go, ' +
            'answer the camera prompt first, or confirm the laptop URL on the same Wi‑Fi.',
        );
      } else {
        appendLog(`session error: ${String(e)}`);
      }
      setSessionId(null);
      return null;
    } finally {
      setBusy(false);
    }
  }, [appendLog, baseUrl, destination]);

  const sendOneFrame = useCallback(async () => {
    const base = normalizeBase(baseUrl);
    const sid = sessionId;
    if (!base || !sid || !camRef.current || !cameraReady || inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const photo = await camRef.current.takePictureAsync({
        quality: 0.55,
        skipProcessing: true,
        shutterSound: false,
      });
      const fd = new FormData();
      fd.append('session_id', sid);
      fd.append('image', {
        uri: photo.uri,
        name: 'frame.jpg',
        type: 'image/jpeg',
      } as any);

      const res = await fetchWithTimeout(`${base}/stream/frame`, { method: 'POST', body: fd }, FRAME_TIMEOUT_MS);
      if (!res.ok) {
        appendLog(`frame HTTP ${res.status}`);
        return;
      }
      const j = (await res.json()) as StreamJson;
      const d = j.decision;
      if (j.speak && d?.message) {
        maybeSpeak(d.message, d.priority ?? 0);
      }
      if (d?.action && d.action !== 'silent') {
        appendLog(`${d.action} p=${d.priority ?? 0} ${(d.message ?? '').slice(0, 72)}`);
      }
    } catch (e) {
      if (isAbortError(e)) {
        appendLog('frame error: timed out (first YOLO frame can be very slow).');
      } else {
        appendLog(`frame error: ${String(e)}`);
      }
    } finally {
      inFlightRef.current = false;
    }
  }, [appendLog, baseUrl, cameraReady, maybeSpeak, sessionId]);

  useEffect(() => {
    if (!streaming || !cameraReady) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    intervalRef.current = setInterval(() => {
      void sendOneFrame();
    }, 650);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [sendOneFrame, streaming, cameraReady]);

  const startStreaming = async () => {
    if (streaming || startLockRef.current || busy || streamStarting) return;
    startLockRef.current = true;
    setStreamStarting(true);
    let sid = sessionId;
    try {
      // Ask for camera before hitting the network. On iOS, a pending permission / Local Network
      // sheet can stall fetch(); our old 15s session timeout then surfaced as "aborted".
      if (!permission?.granted) {
        appendLog('Requesting camera permission…');
        const r = await requestPermission();
        if (!r.granted) {
          appendLog('Camera permission denied. Enable it in system Settings → Expo Go → Camera.');
          return;
        }
      }

      if (!sid) {
        appendLog('Creating session (may wait for Local Network prompt on iPhone)…');
        sid = await createSessionRequest();
        if (!sid) return;
      }

      setCameraReady(false);
      setStreaming(true);
      appendLog('camera starting…');
    } finally {
      startLockRef.current = false;
      setStreamStarting(false);
    }
  };

  const stopStreaming = () => {
    setStreaming(false);
    setCameraReady(false);
    Speech.stop();
    appendLog('streaming off');
  };

  const controlsLocked = busy || pinging || streamStarting;

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar style="light" />
      <ScrollView
        contentContainerStyle={styles.scroll}
        keyboardShouldPersistTaps="always"
        nestedScrollEnabled
      >
        <Text style={styles.title}>Assistive Nav stream</Text>
        <Text style={styles.warn}>
          If Expo Go sits on the splash screen forever, your phone cannot reach Metro. The QR URL must use an
          IP your phone can route to (usually 192.168.x.x). If it shows 172.x / 10.x from VPN or Docker, stop the
          dev server and run:{' '}
          <Text style={styles.mono}>npm run start:tunnel</Text>
        </Text>
        {Platform.OS === 'ios' && Constants.appOwnership === 'expo' ? (
          <Text style={styles.critical}>
            Stream server on iPhone + Expo Go: use an <Text style={styles.mono}>https://</Text> URL from your Mac,
            e.g. <Text style={styles.mono}>cloudflared tunnel --url http://127.0.0.1:8765</Text> (see repo{' '}
            <Text style={styles.mono}>scripts/tunnel_stream_server.sh</Text>) and paste the{' '}
            <Text style={styles.mono}>trycloudflare.com</Text> host. Plain <Text style={styles.mono}>http://192.168…</Text>{' '}
            usually times out here.
          </Text>
        ) : null}
        <Text style={styles.hint}>
          Navigation server: <Text style={styles.mono}>stream_server.py</Text>. Tap Ping first.{' '}
          <Text style={styles.bold}>Start stream</Text> asks for camera, then opens the session (ngrok HTTPS is
          the reliable path on iOS Expo Go).
        </Text>

        <Text style={styles.label}>Server base URL (stream_server)</Text>
        <TextInput
          style={styles.input}
          value={baseUrl}
          onChangeText={setBaseUrl}
          autoCapitalize="none"
          autoCorrect={false}
          placeholder="http://192.168.x.x:8765"
          placeholderTextColor="#666"
        />

        <Text style={styles.label}>Destination</Text>
        <TextInput
          style={styles.input}
          value={destination}
          onChangeText={setDestination}
          placeholder="Library"
          placeholderTextColor="#666"
        />

        <View style={styles.row}>
          <Pressable
            style={({ pressed }) => [
              styles.actionBtn,
              styles.actionSecondary,
              controlsLocked && styles.actionDisabled,
              pressed && styles.actionPressed,
            ]}
            onPress={() => void pingServer()}
            disabled={controlsLocked}
            hitSlop={8}
          >
            <Text style={styles.actionLabel}>Ping server</Text>
          </Pressable>
          <Pressable
            style={({ pressed }) => [
              styles.actionBtn,
              styles.actionSecondary,
              controlsLocked && styles.actionDisabled,
              pressed && styles.actionPressed,
            ]}
            onPress={() => void createSessionRequest()}
            disabled={controlsLocked}
            hitSlop={8}
          >
            <Text style={styles.actionLabel}>Create session</Text>
          </Pressable>
        </View>

        <Pressable
          style={({ pressed }) => [
            styles.actionBtn,
            styles.actionPrimary,
            styles.startBtn,
            (streaming || controlsLocked) && styles.actionDisabled,
            pressed && styles.actionPressed,
          ]}
          onPress={() => void startStreaming()}
          disabled={streaming || controlsLocked}
          hitSlop={12}
        >
          <Text style={styles.actionLabel}>
            {streaming ? 'Streaming…' : streamStarting ? 'Starting…' : 'Start stream'}
          </Text>
        </Pressable>

        {streaming ? (
          <Pressable
            style={({ pressed }) => [styles.actionBtn, styles.actionDanger, pressed && styles.actionPressed]}
            onPress={stopStreaming}
            hitSlop={12}
          >
            <Text style={styles.actionLabel}>Stop</Text>
          </Pressable>
        ) : null}

        {busy || pinging ? <ActivityIndicator color="#6af" style={{ marginTop: 12 }} /> : null}

        <View style={styles.camBox}>
          {!streaming ? (
            <Text style={styles.camFallback}>Camera starts only after “Start stream” (faster Expo Go load).</Text>
          ) : permission?.granted === false && !permission.canAskAgain ? (
            <Text style={styles.camFallback}>Camera blocked in settings.</Text>
          ) : (
            <CameraView
              ref={camRef}
              style={styles.camera}
              facing="back"
              onCameraReady={() => setCameraReady(true)}
            />
          )}
        </View>
        <Text style={styles.camMeta}>
          {streaming ? (cameraReady ? 'Camera ready · sending frames' : 'Opening camera…') : 'Camera idle'} ·
          session {sessionId ? 'yes' : 'no'}
        </Text>

        <Text style={styles.label}>Log</Text>
        <View style={styles.logBox}>
          {log.map((line, i) => (
            <Text key={i} style={styles.logLine}>
              {line}
            </Text>
          ))}
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#111' },
  scroll: { padding: 16, paddingBottom: 40 },
  title: { color: '#fff', fontSize: 20, fontWeight: '700', marginBottom: 8 },
  warn: {
    color: '#fc6',
    fontSize: 12,
    marginBottom: 10,
    lineHeight: 17,
    backgroundColor: '#2a2218',
    padding: 10,
    borderRadius: 8,
  },
  critical: {
    color: '#ffb4b4',
    fontSize: 12,
    marginBottom: 12,
    lineHeight: 17,
    backgroundColor: '#3a1818',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#633',
  },
  mono: { fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }) },
  bold: { fontWeight: '700', color: '#ccc' },
  hint: { color: '#999', fontSize: 13, marginBottom: 16, lineHeight: 18 },
  label: { color: '#888', fontSize: 12, marginTop: 8 },
  input: {
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
    marginTop: 4,
    color: '#fff',
    backgroundColor: '#1a1a1a',
  },
  row: { flexDirection: 'row', gap: 10, marginTop: 16 },
  actionBtn: {
    flex: 1,
    paddingVertical: 14,
    paddingHorizontal: 8,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 50,
  },
  actionPrimary: { backgroundColor: '#2a6bfd' },
  actionSecondary: { backgroundColor: '#2c2c3e' },
  actionDanger: { backgroundColor: '#a33', marginTop: 10 },
  actionDisabled: { opacity: 0.4 },
  actionPressed: { opacity: 0.85 },
  startBtn: { marginTop: 12, alignSelf: 'stretch' },
  actionLabel: { color: '#fff', fontWeight: '700', fontSize: 15 },
  camBox: {
    marginTop: 16,
    height: 220,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  camera: { flex: 1 },
  camFallback: { color: '#888', padding: 16, textAlign: 'center', marginTop: 72, paddingHorizontal: 12 },
  camMeta: { color: '#666', fontSize: 11, marginTop: 6 },
  logBox: {
    marginTop: 6,
    padding: 10,
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#292929',
  },
  logLine: { color: '#aaa', fontSize: 11, marginBottom: 4 },
});
