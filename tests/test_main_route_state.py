from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import NavigationApp, SharedRouteContext, _parse_origin_lon_lat  # noqa: E402


def test_parse_origin_lon_lat_accepts_valid_coordinate_pair() -> None:
    assert _parse_origin_lon_lat("-118.448,34.070") == (-118.448, 34.070)


def test_shared_route_context_exposes_latest_instruction() -> None:
    route = SharedRouteContext("Rieber Hall")
    assert route.get().active is False

    route.update_instruction("Walk straight")
    state = route.get()

    assert state.active is True
    assert state.destination == "Rieber Hall"
    assert state.next_instruction == "Walk straight"


def test_route_builder_can_force_mock_route_without_origin() -> None:
    app = NavigationApp()
    route, repeat, source = app._build_route(
        Namespace(route_provider="mock", origin=None),
        "Rieber Hall",
    )

    assert route is None
    assert repeat is True
    assert source == "mock"


def main() -> None:
    test_parse_origin_lon_lat_accepts_valid_coordinate_pair()
    test_shared_route_context_exposes_latest_instruction()
    test_route_builder_can_force_mock_route_without_origin()
    print("main route state tests passed")


if __name__ == "__main__":
    main()
