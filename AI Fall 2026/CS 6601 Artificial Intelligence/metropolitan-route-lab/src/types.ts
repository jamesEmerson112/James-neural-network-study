export type AlgorithmId =
  | "dijkstra"
  | "bidijkstra"
  | "astar"
  | "alt"
  | "reach"
  | "reachshort"
  | "combo";

export interface GraphNode {
  id: number;
  osmId: number;
  lat: number;
  lon: number;
  junction: boolean;
}

export interface GraphEdge {
  id: number;
  from: number;
  to: number;
  weight: number;
  distance: number;
  speed: number;
  highway: string;
  name: string;
  wayId: number;
  path: number[];
}

export interface RoadGraphData {
  meta: {
    name: string;
    generatedAt: string;
    osmTimestamp: string;
    attribution: string;
    query: string;
    bounds: [number, number, number, number];
    maxSpeedKph: number;
    rawWayCount: number;
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
  shortcuts: GraphEdge[];
}

export interface RuntimeEdge extends GraphEdge {
  shortcut: boolean;
}

export interface RuntimeGraph extends RoadGraphData {
  adj: RuntimeEdge[][];
  rev: RuntimeEdge[][];
  shortcutAdj: RuntimeEdge[][];
  shortcutRev: RuntimeEdge[][];
  selectableNodes: number[];
}

export interface SearchEvent {
  node: number;
  side: "forward" | "backward";
  g: number;
  h: number;
  priority: number;
  frontier: number;
  pruned: boolean;
  viaShortcut: boolean;
}

export interface SearchResult {
  requestId: number;
  algorithm: AlgorithmId;
  source: number;
  target: number;
  cost: number;
  distance: number;
  route: number[];
  events: SearchEvent[];
  expanded: number;
  relaxed: number;
  pruned: number;
  frontierPeak: number;
  queryMs: number;
  memoryBytes: number;
  verified: boolean;
  oracleCost: number;
}

export interface PreprocessData {
  landmarkIds: number[];
  landmarkFrom: number[][];
  landmarkTo: number[][];
  reach: number[];
  landmarkMs: number;
  reachMs: number;
  graphBuildMs: number;
  indexBytes: number;
  fromCache?: boolean;
}

export type WorkerRequest =
  | { type: "initialize"; graph: RoadGraphData; cached?: PreprocessData }
  | { type: "rebuild" }
  | { type: "search"; requestId: number; algorithm: AlgorithmId; source: number; target: number }
  | { type: "race"; requestId: number; source: number; target: number };

export type WorkerResponse =
  | { type: "progress"; phase: string; current: number; total: number }
  | { type: "ready"; data: PreprocessData }
  | { type: "result"; result: SearchResult }
  | { type: "race-result"; requestId: number; results: SearchResult[] }
  | { type: "error"; message: string };
