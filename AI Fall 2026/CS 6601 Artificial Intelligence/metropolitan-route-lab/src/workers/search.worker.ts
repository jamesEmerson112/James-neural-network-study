/// <reference lib="webworker" />

import { buildRuntimeGraph } from "../data/runtimeGraph";
import { preprocessGraph, runSearch } from "../algorithms/search";
import type { AlgorithmId, PreprocessData, RuntimeGraph, WorkerRequest, WorkerResponse } from "../types";

const context = self as DedicatedWorkerGlobalScope;
let graph: RuntimeGraph | undefined;
let indexes: PreprocessData | undefined;
let graphBuildMs = 0;

function send(message: WorkerResponse): void {
  context.postMessage(message);
}

async function buildIndexes(cached?: PreprocessData): Promise<void> {
  if (!graph) return;
  if (cached && cached.reach.length === graph.nodes.length) {
    indexes = { ...cached, fromCache: true, graphBuildMs };
  } else {
    indexes = await preprocessGraph(graph, graphBuildMs, (phase, current, total) => {
      send({ type: "progress", phase, current, total });
    });
  }
  send({ type: "ready", data: indexes });
}

context.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  try {
    const message = event.data;
    if (message.type === "initialize") {
      const started = performance.now();
      graph = buildRuntimeGraph(message.graph);
      graphBuildMs = performance.now() - started;
      await buildIndexes(message.cached);
      return;
    }
    if (message.type === "rebuild") {
      indexes = undefined;
      await buildIndexes();
      return;
    }
    if (!graph || !indexes) throw new Error("The road graph is still preprocessing.");
    if (message.type === "search") {
      send({ type: "result", result: runSearch(graph, indexes, message.requestId, message.algorithm, message.source, message.target) });
      return;
    }
    if (message.type === "race") {
      const algorithms: AlgorithmId[] = ["dijkstra", "bidijkstra", "astar", "alt", "reach", "reachshort", "combo"];
      const oracle = runSearch(graph, indexes, message.requestId, "dijkstra", message.source, message.target);
      const results = algorithms.map((algorithm) => algorithm === "dijkstra"
        ? oracle
        : runSearch(graph!, indexes!, message.requestId, algorithm, message.source, message.target, oracle.cost));
      send({ type: "race-result", requestId: message.requestId, results });
    }
  } catch (error) {
    send({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};

export {};
