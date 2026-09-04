import type { RoadGraphData, RuntimeEdge, RuntimeGraph } from "../types";

export function buildRuntimeGraph(data: RoadGraphData): RuntimeGraph {
  const n = data.nodes.length;
  const adj: RuntimeEdge[][] = Array.from({ length: n }, () => []);
  const rev: RuntimeEdge[][] = Array.from({ length: n }, () => []);
  const shortcutAdj: RuntimeEdge[][] = Array.from({ length: n }, () => []);
  const shortcutRev: RuntimeEdge[][] = Array.from({ length: n }, () => []);

  const attach = (edge: RuntimeEdge, forward: RuntimeEdge[][], backward: RuntimeEdge[][]) => {
    forward[edge.from].push(edge);
    backward[edge.to].push({ ...edge, from: edge.to, to: edge.from, path: [...edge.path].reverse() });
  };

  data.edges.forEach((edge) => attach({ ...edge, shortcut: false }, adj, rev));
  data.shortcuts.forEach((edge) => attach({ ...edge, shortcut: edge.path.length > 2 }, shortcutAdj, shortcutRev));

  return {
    ...data,
    adj,
    rev,
    shortcutAdj,
    shortcutRev,
    selectableNodes: data.nodes.filter((node) => node.junction).map((node) => node.id)
  };
}
