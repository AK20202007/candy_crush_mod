from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class LocationFix:
    latitude: float
    longitude: float
    horizontal_accuracy_m: Optional[float] = None
    source: str = "macos-corelocation"

    def as_lon_lat(self) -> tuple[float, float]:
        return (self.longitude, self.latitude)


def _get_ip_location_fallback() -> LocationFix:
    """Fallback to IP-based location if CoreLocation fails or is disabled."""
    try:
        req = urllib.request.Request("http://ip-api.com/json/", method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                return LocationFix(
                    latitude=float(data["lat"]),
                    longitude=float(data["lon"]),
                    source="ip-api"
                )
    except Exception as e:
        print(f"[location_service] IP fallback failed: {e}")
    raise RuntimeError("Could not determine location from GPS or IP fallback.")


def get_current_location(timeout_s: float = 8.0) -> LocationFix:
    """Return the current macOS Location Services fix.

    This requires macOS Location Services permission for the terminal/Python app.
    The function falls back to IP location on denial, timeout, or missing PyObjC bindings.
    """
    try:
        import CoreLocation
        import objc
        from Foundation import NSDate, NSObject, NSRunLoop
    except Exception as exc:
        print(f"[location_service] CoreLocation bindings unavailable ({exc}). Falling back to IP location.")
        return _get_ip_location_fallback()

    class _Delegate(NSObject):
        fix = None
        error = None
        done = False

        def locationManager_didUpdateLocations_(self, manager, locations):  # noqa: N802
            if not locations:
                return
            loc = locations[-1]
            coord = loc.coordinate()
            self.fix = LocationFix(
                latitude=float(coord.latitude),
                longitude=float(coord.longitude),
                horizontal_accuracy_m=float(loc.horizontalAccuracy()),
            )
            self.done = True
            manager.stopUpdatingLocation()

        def locationManager_didFailWithError_(self, manager, error):  # noqa: N802
            self.error = str(error)
            self.done = True
            manager.stopUpdatingLocation()

        def locationManagerDidChangeAuthorization_(self, manager):  # noqa: N802
            status = manager.authorizationStatus()
            denied = {
                getattr(CoreLocation, "kCLAuthorizationStatusDenied", 2),
                getattr(CoreLocation, "kCLAuthorizationStatusRestricted", 1),
            }
            if status in denied:
                self.error = "Location Services permission was denied or restricted."
                self.done = True

    manager = CoreLocation.CLLocationManager.alloc().init()
    delegate = _Delegate.alloc().init()
    manager.setDelegate_(delegate)

    if hasattr(manager, "setDesiredAccuracy_"):
        manager.setDesiredAccuracy_(CoreLocation.kCLLocationAccuracyBest)

    services_enabled = bool(CoreLocation.CLLocationManager.locationServicesEnabled())
    if not services_enabled:
        print("[location_service] macOS Location Services is turned off. Falling back to IP location.")
        return _get_ip_location_fallback()

    status = CoreLocation.CLLocationManager.authorizationStatus()
    denied_statuses = {
        getattr(CoreLocation, "kCLAuthorizationStatusDenied", 2),
        getattr(CoreLocation, "kCLAuthorizationStatusRestricted", 1),
    }
    if status in denied_statuses:
        print("[location_service] Location Services permission is denied. Falling back to IP location.")
        return _get_ip_location_fallback()

    not_determined = getattr(CoreLocation, "kCLAuthorizationStatusNotDetermined", 0)
    if status == not_determined and hasattr(manager, "requestWhenInUseAuthorization"):
        manager.requestWhenInUseAuthorization()

    manager.startUpdatingLocation()

    deadline = time.time() + max(1.0, timeout_s)
    run_loop = NSRunLoop.currentRunLoop()
    while time.time() < deadline and not delegate.done:
        date = NSDate.dateWithTimeIntervalSinceNow_(0.1)
        run_loop.runUntilDate_(date)

    manager.stopUpdatingLocation()
    manager.setDelegate_(None)
    objc.removeAutoreleasePool()

    if delegate.fix is not None:
        return delegate.fix
    if delegate.error:
        print(f"[location_service] GPS error: {delegate.error}. Falling back to IP location.")
        return _get_ip_location_fallback()
    print("[location_service] Timed out waiting for macOS location fix. Falling back to IP location.")
    return _get_ip_location_fallback()


def is_indoors(lat: float, lon: float, timeout_s: float = 10.0) -> bool:
    """Check if the given coordinate is inside a building footprint using OSM Overpass API.
    
    A 15-meter radius is used to compensate for typical GPS inaccuracies.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Query for any way or relation with building=* tag within 15 meters
    query = f"""
    [out:json][timeout:{int(timeout_s)}];
    (
      way["building"](around:15.0, {lat}, {lon});
      relation["building"](around:15.0, {lat}, {lon});
    );
    out tags;
    """
    
    req = urllib.request.Request(
        overpass_url,
        data=query.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
            elements = data.get("elements", [])
            # If any building element is found in the vicinity, we assume we are indoors
            # or exiting one.
            return len(elements) > 0
    except Exception as e:
        print(f"[location_service] Overpass API check failed: {e}. Defaulting to outdoors.")
        return False
