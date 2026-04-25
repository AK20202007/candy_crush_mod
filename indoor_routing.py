"""Simple indoor routing using graph-based pathfinding."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class IndoorNode:
    """A node in the indoor graph (room, landmark, corridor junction)."""
    id: str
    name: str
    floor: int
    node_type: str  # room, landmark, junction, entrance, elevator, stairs
    x: float  # relative x coordinate
    y: float  # relative y coordinate
    tags: List[str] = field(default_factory=list)  # e.g., ["restroom", "accessible"]


@dataclass
class IndoorEdge:
    """An edge in the indoor graph (corridor, path)."""
    from_id: str
    to_id: str
    distance_m: float
    edge_type: str = "corridor"  # corridor, stairs, elevator, door
    accessible: bool = True


@dataclass
class IndoorRouteStep:
    """A single step in an indoor route."""
    instruction: str
    distance_m: float
    direction: str  # straight, left, right, back
    landmark: Optional[str] = None


@dataclass
class IndoorRoute:
    """A complete indoor route."""
    steps: List[IndoorRouteStep]
    total_distance_m: float
    floor_transitions: List[Tuple[int, int]]  # (from_floor, to_floor)


class IndoorGraph:
    """Graph-based indoor routing."""

    def __init__(self) -> None:
        self.nodes: Dict[str, IndoorNode] = {}
        self.edges: Dict[str, List[IndoorEdge]] = {}  # from_id -> edges
        self.adjacency: Dict[str, Dict[str, float]] = {}  # from_id -> to_id -> distance

    def add_node(self, node: IndoorNode) -> None:
        self.nodes[node.id] = node
        self.edges[node.id] = []
        self.adjacency[node.id] = {}

    def add_edge(self, edge: IndoorEdge) -> None:
        self.edges[edge.from_id].append(edge)
        self.adjacency[edge.from_id][edge.to_id] = edge.distance_m
        # Undirected graph
        reverse_edge = IndoorEdge(
            from_id=edge.to_id,
            to_id=edge.from_id,
            distance_m=edge.distance_m,
            edge_type=edge.edge_type,
            accessible=edge.accessible,
        )
        self.edges[edge.to_id].append(reverse_edge)
        self.adjacency[edge.to_id][edge.from_id] = edge.distance_m

    def dijkstra(self, start_id: str, end_id: str) -> Tuple[List[str], float]:
        """Find shortest path using Dijkstra's algorithm."""
        if start_id not in self.nodes or end_id not in self.nodes:
            raise ValueError(f"Node not found: {start_id} or {end_id}")

        distances: Dict[str, float] = {node_id: float("inf") for node_id in self.nodes}
        distances[start_id] = 0
        previous: Dict[str, Optional[str]] = {node_id: None for node_id in self.nodes}
        visited: Set[str] = set()

        heap: List[Tuple[float, str]] = [(0, start_id)]

        while heap:
            current_dist, current_id = heapq.heappop(heap)

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == end_id:
                break

            for neighbor_id, edge_dist in self.adjacency[current_id].items():
                if neighbor_id in visited:
                    continue
                new_dist = current_dist + edge_dist
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    previous[neighbor_id] = current_id
                    heapq.heappush(heap, (new_dist, neighbor_id))

        # Reconstruct path
        path: List[str] = []
        current = end_id
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return path, distances[end_id]

    def generate_instructions(self, path: List[str]) -> List[IndoorRouteStep]:
        """Generate human-readable navigation instructions from path."""
        if len(path) < 2:
            return []

        steps: List[IndoorRouteStep] = []
        for i in range(len(path) - 1):
            from_node = self.nodes[path[i]]
            to_node = self.nodes[path[i + 1]]
            
            # Find the edge
            edge = next(
                (e for e in self.edges[from_node.id] if e.to_id == to_node.id),
                None,
            )
            if edge is None:
                continue

            # Calculate direction based on coordinates
            dx = to_node.x - from_node.x
            dy = to_node.y - from_node.y
            direction = self._direction_from_vector(dx, dy)

            # Generate instruction
            instruction = self._generate_step_instruction(
                from_node, to_node, edge, direction, i == 0
            )

            landmark = to_node.name if to_node.node_type == "landmark" else None

            steps.append(
                IndoorRouteStep(
                    instruction=instruction,
                    distance_m=edge.distance_m,
                    direction=direction,
                    landmark=landmark,
                )
            )

        return steps

    def _direction_from_vector(self, dx: float, dy: float) -> str:
        """Convert dx, dy to compass direction."""
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "straight" if dy > 0 else "back"

    def _generate_step_instruction(
        self,
        from_node: IndoorNode,
        to_node: IndoorNode,
        edge: IndoorEdge,
        direction: str,
        is_first: bool,
    ) -> str:
        """Generate a human-readable instruction for a step."""
        if edge.edge_type == "stairs":
            return f"Take the {direction} stairs to {to_node.name}"
        elif edge.edge_type == "elevator":
            return f"Take the elevator to floor {to_node.floor}"
        elif edge.edge_type == "door":
            return f"Go {direction} through the door to {to_node.name}"
        elif to_node.node_type == "landmark":
            return f"Walk {direction} toward {to_node.name}"
        elif to_node.node_type == "junction":
            return f"Continue {direction} at the junction"
        else:
            if is_first:
                return f"Walk {direction} for about {int(edge.distance_m)} meters"
            else:
                return f"Turn {direction} and walk about {int(edge.distance_m)} meters"

    def find_route(self, start_id: str, end_id: str) -> IndoorRoute:
        """Find a complete route between two nodes."""
        path, total_distance = self.dijkstra(start_id, end_id)
        steps = self.generate_instructions(path)

        # Find floor transitions
        floor_transitions: List[Tuple[int, int]] = []
        for i in range(len(path) - 1):
            from_floor = self.nodes[path[i]].floor
            to_floor = self.nodes[path[i + 1]].floor
            if from_floor != to_floor:
                floor_transitions.append((from_floor, to_floor))

        return IndoorRoute(
            steps=steps,
            total_distance_m=total_distance,
            floor_transitions=floor_transitions,
        )


def load_graph_from_json(path: Path) -> IndoorGraph:
    """Load an indoor graph from a JSON file."""
    import json

    with path.open("r") as f:
        data = json.load(f)

    graph = IndoorGraph()

    for node_data in data.get("nodes", []):
        node = IndoorNode(
            id=node_data["id"],
            name=node_data["name"],
            floor=node_data["floor"],
            node_type=node_data["type"],
            x=node_data["x"],
            y=node_data["y"],
            tags=node_data.get("tags", []),
        )
        graph.add_node(node)

    for edge_data in data.get("edges", []):
        edge = IndoorEdge(
            from_id=edge_data["from"],
            to_id=edge_data["to"],
            distance_m=edge_data["distance"],
            edge_type=edge_data.get("type", "corridor"),
            accessible=edge_data.get("accessible", True),
        )
        graph.add_edge(edge)

    return graph


def find_node_by_name(graph: IndoorGraph, name: str) -> Optional[str]:
    """Find a node ID by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for node_id, node in graph.nodes.items():
        if name_lower in node.name.lower():
            return node_id
    return None
