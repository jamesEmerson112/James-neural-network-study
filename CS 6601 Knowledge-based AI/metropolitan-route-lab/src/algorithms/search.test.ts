import { beforeAll, describe, expect, it } from "vitest";
import snapshot from "../data/san-francisco.graph.json";
import { buildRuntimeGraph } from "../data/runtimeGraph";
import { preprocessGraph, runSearch } from "./search";
import type { AlgorithmId, PreprocessData, RoadGraphData, RuntimeGraph } from "../types";

const graphData = snapshot as RoadGraphData;
let graph: RuntimeGraph;
let indexes: PreprocessData;

function nearest(lat: number, lon: number): number {
  let bestId = graph.selectableNodes[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const id of graph.selectableNodes) {
    const node = graph.nodes[id];
    const distance = (node.lat - lat) ** 2 + ((node.lon - lon) * Math.cos(lat * Math.PI / 180)) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestId = id;
    }
  }
  return bestId;
}

beforeAll(async () => {
  graph = buildRuntimeGraph(graphData);
  indexes = await preprocessGraph(graph, 0);
});

describe("real San Francisco routing graph", () => {
  it("contains real directed roads and a smaller exact corridor graph", () => {
    expect(graph.nodes.length).toBeGreaterThan(2000);
    expect(graph.edges.length).toBeGreaterThan(2500);
    expect(graph.shortcuts.length).toBeLessThan(graph.edges.length);
    expect(graph.shortcuts.some((edge) => edge.path.length > 2)).toBe(true);
  });

  it("keeps every heuristic and pruning variant equal to Dijkstra", () => {
    const source = nearest(37.7952, -122.3937);
    const target = nearest(37.7794, -122.4192);
    const methods: AlgorithmId[] = ["dijkstra", "bidijkstra", "astar", "alt", "reach", "reachshort", "combo"];
    const results = methods.map((method, requestId) => runSearch(graph, indexes, requestId, method, source, target));
    expect(results[0].route[0]).toBe(source);
    expect(results[0].route.at(-1)).toBe(target);
    for (const result of results) {
      expect(result.verified, result.algorithm).toBe(true);
      expect(result.cost, result.algorithm).toBeCloseTo(results[0].cost, 8);
      expect(result.route[0], result.algorithm).toBe(source);
      expect(result.route.at(-1), result.algorithm).toBe(target);
    }
  });

  it("matches Dijkstra on three additional directed trips", () => {
    const pairs = [
      [nearest(37.7793, -122.4192), nearest(37.7897, -122.3960)],
      [nearest(37.7765, -122.4100), nearest(37.7940, -122.4078)],
      [nearest(37.7860, -122.4210), nearest(37.7925, -122.3940)]
    ];
    const methods: AlgorithmId[] = ["bidijkstra", "astar", "alt", "reach", "reachshort", "combo"];
    for (const [source, target] of pairs) {
      const oracle = runSearch(graph, indexes, 0, "dijkstra", source, target);
      expect(Number.isFinite(oracle.cost)).toBe(true);
      for (const method of methods) {
        const result = runSearch(graph, indexes, 0, method, source, target, oracle.cost);
        expect(result.verified, `${method} on ${source}→${target}`).toBe(true);
      }
    }
  });

  it("differentially checks optimized methods on deterministic graph-wide pairs", () => {
    const methods: AlgorithmId[] = ["bidijkstra", "astar", "alt", "reach", "reachshort", "combo"];
    for (let index = 0; index < 32; index += 1) {
      const source = graph.selectableNodes[(index * 37 + 11) % graph.selectableNodes.length];
      const target = graph.selectableNodes[(index * 101 + 73) % graph.selectableNodes.length];
      if (source === target) continue;
      const oracle = runSearch(graph, indexes, index, "dijkstra", source, target);
      for (const method of methods) {
        const result = runSearch(graph, indexes, index, method, source, target, oracle.cost);
        expect(result.verified, `${method} on ${source}→${target}`).toBe(true);
      }
    }
  });
});
