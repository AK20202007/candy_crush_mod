declare module "expo-camera" {
  import * as React from "react";
  import { ViewProps } from "react-native";

  export type CameraPermissionResponse = {
    granted: boolean;
    canAskAgain: boolean;
    expires: "never" | number;
    status: string;
  };

  export function useCameraPermissions(): [
    CameraPermissionResponse | null,
    () => Promise<CameraPermissionResponse>
  ];

  export const CameraView: React.ComponentType<
    ViewProps & {
      facing?: "front" | "back";
      barcodeScannerSettings?: { barcodeTypes?: string[] };
      onBarcodeScanned?: (event: { type?: string; data?: string }) => void;
    }
  >;
}
