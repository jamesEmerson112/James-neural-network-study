import snapshot from "../src/data/san-francisco.graph.json";
import { buildRuntimeGraph } from "../src/data/runtimeGraph";
import { preprocessGraph, runSearch } from "../src/algorithms/search";
import type { AlgorithmId, RoadGraphData } from "../src/types";

const graph = buildRuntimeGraph(snapshot as RoadGraphData);
const indexes = await preprocessGraph(graph, 0);

function nearest(lat: number, lon: number): number {
  return graph.selectableNodes.reduce((best, id) => {
    const node = graph.nodes[id];
    const prior = graph.nodes[best];
    const score = (node.lat - lat) ** 2 + (node.lon - lon) ** 2;
    const priorScore = (prior.lat - lat) ** 2 + (prior.lon - lon) ** 2;
    return score < priorScore ? id : best;
  }, graph.selectableNodes[0]);
}

const source = nearest(37.7952, -122.3937);
const target = nearest(37.7794, -122.4192);
const algorithms: AlgorithmId[] = ["dijkstra", "bidijkstra", "astar", "alt", "reach", "reachshort", "combo"];
const oracle = runSearch(graph, indexes, 0, "dijkstra", source, target);
const rows = algorithms.map((algorithm) => {
  const result = algorithm === "dijkstra" ? oracle : runSearch(graph, indexes, 0, algorithm, source, target, oracle.cost);
  return {
    algorithm,
    exact: result.verified,
    minutes: Number(result.cost.toFixed(4)),
    settled: result.expanded,
    relaxed: result.relaxed,
    pruned: result.pruned,
    "worker ms": Number(result.queryMs.toFixed(3))
  };
});

console.log(`${graph.meta.name}: ${graph.nodes.length} nodes, ${graph.edges.length} directed arcs`);
console.log(`Preprocessing: landmarks ${indexes.landmarkMs.toFixed(1)} ms, reach ${indexes.reachMs.toFixed(1)} ms`);
console.table(rows);
