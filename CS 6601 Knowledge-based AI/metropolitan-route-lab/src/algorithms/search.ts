import { MinHeap } from "./heap";
import type {
  AlgorithmId,
  PreprocessData,
  RuntimeEdge,
  RuntimeGraph,
  SearchEvent,
  SearchResult
} from "../types";

const EPSILON = 1e-8;

function filled(length: number, value: number): Float64Array {
  const result = new Float64Array(length);
  result.fill(value);
  return result;
}

function dijkstraDistances(adjacency: RuntimeEdge[][], source: number): Float64Array {
  const distance = filled(adjacency.length, Number.POSITIVE_INFINITY);
  const heap = new MinHeap();
  distance[source] = 0;
  heap.push({ node: source, g: 0, key: 0 });
  while (heap.size) {
    const item = heap.pop()!;
    if (Math.abs(item.g - distance[item.node]) > EPSILON) continue;
    for (const edge of adjacency[item.node]) {
      const next = item.g + edge.weight;
      if (next + EPSILON < distance[edge.to]) {
        distance[edge.to] = next;
        heap.push({ node: edge.to, g: next, key: next });
      }
    }
  }
  return distance;
}

function chooseLandmarks(graph: RuntimeGraph, count = 8): number[] {
  const candidates = graph.selectableNodes.length ? graph.selectableNodes : graph.nodes.map((node) => node.id);
  if (!candidates.length) return [];
  const extrema = [
    (id: number) => graph.nodes[id].lat,
    (id: number) => -graph.nodes[id].lat,
    (id: number) => graph.nodes[id].lon,
    (id: number) => -graph.nodes[id].lon,
    (id: number) => graph.nodes[id].lat + graph.nodes[id].lon,
    (id: number) => graph.nodes[id].lat - graph.nodes[id].lon,
    (id: number) => -graph.nodes[id].lat + graph.nodes[id].lon,
    (id: number) => -graph.nodes[id].lat - graph.nodes[id].lon
  ];
  const result: number[] = [];
  for (const score of extrema) {
    const selected = candidates.reduce((best, id) => score(id) > score(best) ? id : best, candidates[0]);
    if (!result.includes(selected)) result.push(selected);
    if (result.length === count) break;
  }
  return result;
}

export async function preprocessGraph(
  graph: RuntimeGraph,
  graphBuildMs: number,
  progress?: (phase: string, current: number, total: number) => void
): Promise<PreprocessData> {
  const landmarkStarted = performance.now();
  const landmarkIds = chooseLandmarks(graph);
  const landmarkFrom: number[][] = [];
  const landmarkTo: number[][] = [];
  for (let index = 0; index < landmarkIds.length; index += 1) {
    landmarkFrom.push(Array.from(dijkstraDistances(graph.adj, landmarkIds[index])));
    landmarkTo.push(Array.from(dijkstraDistances(graph.rev, landmarkIds[index])));
    progress?.("Landmark distances", index + 1, landmarkIds.length);
    await Promise.resolve();
  }
  const landmarkMs = performance.now() - landmarkStarted;

  // Exact vertex reach on the imported graph. For each source, the shortest-path
  // DAG tells us the furthest destination whose shortest route can pass through v.
  // Processing vertices in reverse distance order avoids an O(V^3) scan.
  const reachStarted = performance.now();
  const reach = new Float64Array(graph.nodes.length);
  const order = new Array<number>(graph.nodes.length);
  const furthest = new Float64Array(graph.nodes.length);
  for (let source = 0; source < graph.nodes.length; source += 1) {
    const distance = dijkstraDistances(graph.adj, source);
    for (let i = 0; i < order.length; i += 1) {
      order[i] = i;
      furthest[i] = distance[i];
    }
    order.sort((a, b) => distance[b] - distance[a]);
    for (const vertex of order) {
      if (!Number.isFinite(distance[vertex])) continue;
      for (const edge of graph.adj[vertex]) {
        if (!Number.isFinite(distance[edge.to])) continue;
        const tolerance = EPSILON * Math.max(1, distance[edge.to]);
        if (Math.abs(distance[vertex] + edge.weight - distance[edge.to]) <= tolerance) {
          furthest[vertex] = Math.max(furthest[vertex], furthest[edge.to]);
        }
      }
      const candidate = Math.min(distance[vertex], furthest[vertex] - distance[vertex]);
      if (candidate > reach[vertex]) reach[vertex] = candidate;
    }
    if (source % 10 === 0 || source === graph.nodes.length - 1) {
      progress?.("Exact reach labels", source + 1, graph.nodes.length);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }
  const reachMs = performance.now() - reachStarted;
  const indexBytes = landmarkIds.length * graph.nodes.length * 2 * 8 + reach.byteLength;
  return {
    landmarkIds,
    landmarkFrom,
    landmarkTo,
    reach: Array.from(reach),
    landmarkMs,
    reachMs,
    graphBuildMs,
    indexBytes
  };
}

function haversineMeters(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const radians = Math.PI / 180;
  const dLat = (bLat - aLat) * radians;
  const dLon = (bLon - aLon) * radians;
  const lat1 = aLat * radians;
  const lat2 = bLat * radians;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 6_371_000 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

function straightLineMinutes(graph: RuntimeGraph, from: number, to: number): number {
  const a = graph.nodes[from];
  const b = graph.nodes[to];
  return haversineMeters(a.lat, a.lon, b.lat, b.lon) / (graph.meta.maxSpeedKph * 1000) * 60;
}

function landmarkHeuristic(index: PreprocessData, vertex: number, target: number): number {
  let best = 0;
  for (let i = 0; i < index.landmarkIds.length; i += 1) {
    const from = index.landmarkFrom[i];
    const to = index.landmarkTo[i];
    const forward = from[target] - from[vertex];
    const reverse = to[vertex] - to[target];
    if (Number.isFinite(forward)) best = Math.max(best, forward);
    if (Number.isFinite(reverse)) best = Math.max(best, reverse);
  }
  return Math.max(0, best);
}

function expandEdgePath(route: number[], edge: RuntimeEdge): void {
  const path = edge.path.length ? edge.path : [edge.from, edge.to];
  for (let i = 1; i < path.length; i += 1) route.push(path[i]);
}

function reconstruct(parent: Array<RuntimeEdge | undefined>, source: number, target: number): number[] {
  const edges: RuntimeEdge[] = [];
  let cursor = target;
  let guard = parent.length + 1;
  while (cursor !== source && guard-- > 0) {
    const edge = parent[cursor];
    if (!edge) return [];
    edges.push(edge);
    cursor = edge.from;
  }
  edges.reverse();
  const route = [source];
  edges.forEach((edge) => expandEdgePath(route, edge));
  return route;
}

function reverseEdge(edge: RuntimeEdge): RuntimeEdge {
  return { ...edge, from: edge.to, to: edge.from, path: [...edge.path].reverse() };
}

interface RawResult extends Omit<SearchResult, "requestId" | "queryMs" | "verified" | "oracleCost" | "distance"> {}

function oneWaySearch(
  graph: RuntimeGraph,
  index: PreprocessData,
  algorithm: AlgorithmId,
  source: number,
  target: number
): RawResult {
  const usesShortcuts = algorithm === "combo";
  const usesReach = algorithm === "combo";
  const adjacency = usesShortcuts ? graph.shortcutAdj : graph.adj;
  const n = graph.nodes.length;
  const distance = filled(n, Number.POSITIVE_INFINITY);
  const closed = new Uint8Array(n);
  const parent = new Array<RuntimeEdge | undefined>(n);
  const events: SearchEvent[] = [];
  const heap = new MinHeap();
  let relaxed = 0;
  let pruned = 0;
  let frontierPeak = 1;
  const heuristic = (vertex: number) => {
    if (algorithm === "alt" || algorithm === "combo") return landmarkHeuristic(index, vertex, target);
    if (algorithm === "astar") return straightLineMinutes(graph, vertex, target);
    return 0;
  };
  distance[source] = 0;
  heap.push({ node: source, g: 0, key: heuristic(source) });
  while (heap.size) {
    const item = heap.pop()!;
    if (closed[item.node] || Math.abs(item.g - distance[item.node]) > EPSILON) continue;
    const vertex = item.node;
    closed[vertex] = 1;
    const h = heuristic(vertex);
    const lowerBound = usesReach ? Math.max(straightLineMinutes(graph, vertex, target), h) : h;
    const isPruned = usesReach && vertex !== source && vertex !== target &&
      index.reach[vertex] + EPSILON < Math.min(distance[vertex], lowerBound);
    events.push({
      node: vertex,
      side: "forward",
      g: distance[vertex],
      h,
      priority: distance[vertex] + h,
      frontier: heap.size,
      pruned: isPruned,
      viaShortcut: Boolean(parent[vertex]?.shortcut)
    });
    if (isPruned) {
      pruned += 1;
      continue;
    }
    if (vertex === target) break;
    for (const edge of adjacency[vertex]) {
      relaxed += 1;
      const next = distance[vertex] + edge.weight;
      if (next + EPSILON < distance[edge.to]) {
        distance[edge.to] = next;
        parent[edge.to] = edge;
        heap.push({ node: edge.to, g: next, key: next + heuristic(edge.to) });
      }
    }
    frontierPeak = Math.max(frontierPeak, heap.size);
  }
  return {
    algorithm,
    source,
    target,
    cost: distance[target],
    route: Number.isFinite(distance[target]) ? reconstruct(parent, source, target) : [],
    events,
    expanded: events.length,
    relaxed,
    pruned,
    frontierPeak,
    memoryBytes: n * 17 + frontierPeak * 24
  };
}

function cleanHeap(heap: MinHeap, distance: Float64Array, closed: Uint8Array) {
  while (heap.size) {
    const top = heap.peek()!;
    if (!closed[top.node] && Math.abs(top.g - distance[top.node]) <= EPSILON) return top;
    heap.pop();
  }
  return undefined;
}

function bidirectionalSearch(
  graph: RuntimeGraph,
  index: PreprocessData,
  algorithm: AlgorithmId,
  source: number,
  target: number
): RawResult {
  const usesReach = algorithm === "reach" || algorithm === "reachshort";
  const usesShortcuts = algorithm === "reachshort";
  const adj = usesShortcuts ? graph.shortcutAdj : graph.adj;
  const rev = usesShortcuts ? graph.shortcutRev : graph.rev;
  const n = graph.nodes.length;
  const distF = filled(n, Number.POSITIVE_INFINITY);
  const distB = filled(n, Number.POSITIVE_INFINITY);
  const closedF = new Uint8Array(n);
  const closedB = new Uint8Array(n);
  const parentF = new Array<RuntimeEdge | undefined>(n);
  const nextB = new Array<RuntimeEdge | undefined>(n);
  const heapF = new MinHeap();
  const heapB = new MinHeap();
  const events: SearchEvent[] = [];
  let best = Number.POSITIVE_INFINITY;
  let meeting = -1;
  let relaxed = 0;
  let pruned = 0;
  let frontierPeak = 2;
  distF[source] = 0;
  distB[target] = 0;
  heapF.push({ node: source, g: 0, key: 0 });
  heapB.push({ node: target, g: 0, key: 0 });

  const consider = (vertex: number) => {
    const candidate = distF[vertex] + distB[vertex];
    if (candidate < best) {
      best = candidate;
      meeting = vertex;
    }
  };

  while (heapF.size && heapB.size) {
    const topF = cleanHeap(heapF, distF, closedF);
    const topB = cleanHeap(heapB, distB, closedB);
    if (!topF || !topB || topF.key + topB.key >= best - EPSILON) break;
    const forward = topF.key <= topB.key;
    const heap = forward ? heapF : heapB;
    const distance = forward ? distF : distB;
    const closed = forward ? closedF : closedB;
    const adjacency = forward ? adj : rev;
    const item = heap.pop()!;
    if (closed[item.node] || Math.abs(item.g - distance[item.node]) > EPSILON) continue;
    const vertex = item.node;
    closed[vertex] = 1;
    consider(vertex);
    const lowerBound = forward
      ? straightLineMinutes(graph, vertex, target)
      : straightLineMinutes(graph, source, vertex);
    const isPruned = usesReach && vertex !== source && vertex !== target &&
      index.reach[vertex] + EPSILON < Math.min(distance[vertex], lowerBound);
    events.push({
      node: vertex,
      side: forward ? "forward" : "backward",
      g: distance[vertex],
      h: lowerBound,
      priority: distance[vertex],
      frontier: heapF.size + heapB.size,
      pruned: isPruned,
      viaShortcut: forward ? Boolean(parentF[vertex]?.shortcut) : Boolean(nextB[vertex]?.shortcut)
    });
    if (isPruned) {
      pruned += 1;
      continue;
    }
    for (const edge of adjacency[vertex]) {
      relaxed += 1;
      const next = distance[vertex] + edge.weight;
      if (next + EPSILON < distance[edge.to]) {
        distance[edge.to] = next;
        heap.push({ node: edge.to, g: next, key: next });
        if (forward) parentF[edge.to] = edge;
        else nextB[edge.to] = reverseEdge(edge);
      }
      consider(edge.to);
    }
    frontierPeak = Math.max(frontierPeak, heapF.size + heapB.size);
  }

  let route: number[] = [];
  if (meeting >= 0 && Number.isFinite(best)) {
    route = reconstruct(parentF, source, meeting);
    if (!route.length && source === meeting) route = [source];
    let cursor = meeting;
    let guard = n + 1;
    while (cursor !== target && guard-- > 0) {
      const edge = nextB[cursor];
      if (!edge) {
        route = [];
        break;
      }
      expandEdgePath(route, edge);
      cursor = edge.to;
    }
  }
  return {
    algorithm,
    source,
    target,
    cost: best,
    route,
    events,
    expanded: events.length,
    relaxed,
    pruned,
    frontierPeak,
    memoryBytes: n * 34 + frontierPeak * 24
  };
}

function routeDistance(graph: RuntimeGraph, route: number[]): number {
  let total = 0;
  for (let i = 0; i + 1 < route.length; i += 1) {
    const edge = graph.adj[route[i]].find((candidate) => candidate.to === route[i + 1]);
    if (edge) total += edge.distance;
  }
  return total;
}

export function runSearch(
  graph: RuntimeGraph,
  index: PreprocessData,
  requestId: number,
  algorithm: AlgorithmId,
  source: number,
  target: number,
  knownOracle?: number
): SearchResult {
  const started = performance.now();
  const raw = algorithm === "bidijkstra" || algorithm === "reach" || algorithm === "reachshort"
    ? bidirectionalSearch(graph, index, algorithm, source, target)
    : oneWaySearch(graph, index, algorithm, source, target);
  const queryMs = performance.now() - started;
  const oracle = knownOracle ?? oneWaySearch(graph, index, "dijkstra", source, target).cost;
  const verified = Number.isFinite(raw.cost) && Number.isFinite(oracle)
    ? Math.abs(raw.cost - oracle) <= EPSILON * Math.max(1, oracle)
    : raw.cost === oracle;
  return {
    ...raw,
    requestId,
    queryMs,
    distance: routeDistance(graph, raw.route),
    verified,
    oracleCost: oracle
  };
}

export const __test__ = { dijkstraDistances };
