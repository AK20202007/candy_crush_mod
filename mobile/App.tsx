import React from "react";
import { StyleSheet, Text, View } from "react-native";

/**
 * Minimal UI for debugging dev-client / Metro. Restore full app from git history when done.
 */
export default function App(): React.JSX.Element {
  return (
    <View style={styles.root} testID="minimal-root">
      <Text style={styles.title}>LAHacks Nav — minimal</Text>
      <Text style={styles.sub}>If you see this, JS loaded from Metro.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#ffffff",
    alignItems: "center",
    justifyContent: "center",
    padding: 24
  },
  title: { fontSize: 20, fontWeight: "700", color: "#111827", marginBottom: 8 },
  sub: { fontSize: 15, color: "#4b5563", textAlign: "center" }
});
