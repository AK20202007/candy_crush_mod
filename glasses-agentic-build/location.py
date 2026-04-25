from __future__ import annotations

import time
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


def get_current_location(timeout_s: float = 8.0) -> LocationFix:
    """Return the current macOS Location Services fix.

    This requires macOS Location Services permission for the terminal/Python app.
    The function raises RuntimeError with a user-readable message on denial,
    timeout, or missing PyObjC bindings.
    """
    try:
        import CoreLocation
        import objc
        from Foundation import NSDate, NSObject, NSRunLoop
    except Exception as exc:
        raise RuntimeError(
            "CoreLocation bindings are unavailable. Install pyobjc-framework-CoreLocation."
        ) from exc

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
        raise RuntimeError("macOS Location Services is turned off.")

    status = CoreLocation.CLLocationManager.authorizationStatus()
    denied_statuses = {
        getattr(CoreLocation, "kCLAuthorizationStatusDenied", 2),
        getattr(CoreLocation, "kCLAuthorizationStatusRestricted", 1),
    }
    if status in denied_statuses:
        raise RuntimeError("Location Services permission is denied for this terminal/Python app.")

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
        raise RuntimeError(delegate.error)
    raise RuntimeError("Timed out waiting for a macOS location fix.")
