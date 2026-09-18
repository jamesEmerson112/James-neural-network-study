import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = resolve(projectRoot, "src/data/san-francisco.graph.json");
// Downtown San Francisco, including Civic Center, SoMa, Union Square, and the
// Ferry Building. Keeping the extract compact makes exact-reach preprocessing
// practical in a browser while still exposing SF's one-ways and street grid.
const bounds = [37.775, -122.423, 37.798, -122.392];
const roadPattern = "^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street)$";
const query = `[out:json][timeout:90];way["highway"~"${roadPattern}"](${bounds.join(",")});out body;>;out skel qt;`;
const endpoint = process.env.OVERPASS_URL || "https://overpass-api.de/api/interpreter";

const response = await fetch(endpoint, {
  method: "POST",
  headers: {
    "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    "user-agent": "James-neural-network-study/metropolitan-route-lab"
  },
  body: new URLSearchParams({ data: query })
});
if (!response.ok) throw new Error(`Overpass returned ${response.status}: ${await response.text()}`);
const osm = await response.json();

const allowed = new Set([
  "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
  "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
  "residential", "living_street"
]);
const nodeByOsmId = new Map(osm.elements.filter((item) => item.type === "node").map((node) => [node.id, node]));
const inside = (node) => node && node.lat >= bounds[0] && node.lat <= bounds[2] && node.lon >= bounds[1] && node.lon <= bounds[3];
const ways = osm.elements.filter((item) => {
  if (item.type !== "way" || !allowed.has(item.tags?.highway)) return false;
  if (item.tags?.area === "yes") return false;
  if (["no", "private"].includes(item.tags?.access)) return false;
  if (["no", "private"].includes(item.tags?.motor_vehicle)) return false;
  return Array.isArray(item.nodes) && item.nodes.length > 1;
});

function haversine(a, b) {
  const radians = Math.PI / 180;
  const dLat = (b.lat - a.lat) * radians;
  const dLon = (b.lon - a.lon) * radians;
  const lat1 = a.lat * radians;
  const lat2 = b.lat * radians;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 6_371_000 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

const fallbackSpeed = {
  motorway: 80, motorway_link: 48, trunk: 64, trunk_link: 48, primary: 48,
  primary_link: 40, secondary: 40, secondary_link: 35, tertiary: 35,
  tertiary_link: 30, unclassified: 30, residential: 30, living_street: 10
};

function speedKph(tags) {
  const raw = String(tags.maxspeed || "").toLowerCase();
  const value = Number.parseFloat(raw);
  if (Number.isFinite(value)) {
    const converted = raw.includes("mph") ? value * 1.609344 : value;
    return Math.max(8, Math.min(130, converted));
  }
  return fallbackSpeed[tags.highway] || 30;
}

function direction(tags) {
  const raw = String(tags.oneway || "").toLowerCase();
  if (raw === "-1" || raw === "reverse") return -1;
  if (["yes", "true", "1"].includes(raw) || tags.junction === "roundabout" || tags.highway === "motorway") return 1;
  return 0;
}

const rawSegments = [];
for (const way of ways) {
  const speed = speedKph(way.tags);
  const oneWay = direction(way.tags);
  for (let index = 0; index + 1 < way.nodes.length; index += 1) {
    const a = nodeByOsmId.get(way.nodes[index]);
    const b = nodeByOsmId.get(way.nodes[index + 1]);
    if (!inside(a) || !inside(b) || a.id === b.id) continue;
    const distance = haversine(a, b);
    if (distance < 0.05) continue;
    const shared = {
      distance,
      speed,
      weight: distance / (speed * 1000) * 60,
      highway: way.tags.highway,
      name: way.tags.name || way.tags.ref || "Unnamed road",
      wayId: way.id
    };
    if (oneWay >= 0) rawSegments.push({ from: a.id, to: b.id, ...shared });
    if (oneWay <= 0) rawSegments.push({ from: b.id, to: a.id, ...shared });
  }
}

// Keep the largest weakly connected road component; tiny disconnected parking
// and construction fragments make poor routing endpoints.
const neighbours = new Map();
const addNeighbour = (a, b) => {
  if (!neighbours.has(a)) neighbours.set(a, new Set());
  neighbours.get(a).add(b);
};
rawSegments.forEach((edge) => {
  addNeighbour(edge.from, edge.to);
  addNeighbour(edge.to, edge.from);
});
let largest = new Set();
const visited = new Set();
for (const start of neighbours.keys()) {
  if (visited.has(start)) continue;
  const component = new Set([start]);
  const queue = [start];
  visited.add(start);
  while (queue.length) {
    const current = queue.pop();
    for (const next of neighbours.get(current) || []) {
      if (visited.has(next)) continue;
      visited.add(next);
      component.add(next);
      queue.push(next);
    }
  }
  if (component.size > largest.size) largest = component;
}

const componentEdges = rawSegments.filter((edge) => largest.has(edge.from) && largest.has(edge.to));
const bestByPair = new Map();
for (const edge of componentEdges) {
  const key = `${edge.from}>${edge.to}`;
  const previous = bestByPair.get(key);
  if (!previous || edge.weight < previous.weight) bestByPair.set(key, edge);
}
const osmIds = [...largest].sort((a, b) => a - b);
const internalId = new Map(osmIds.map((id, index) => [id, index]));

const undirected = new Map(osmIds.map((id) => [id, new Set()]));
const wayUse = new Map(osmIds.map((id) => [id, new Set()]));
for (const edge of bestByPair.values()) {
  undirected.get(edge.from).add(edge.to);
  undirected.get(edge.to).add(edge.from);
  wayUse.get(edge.from).add(edge.wayId);
  wayUse.get(edge.to).add(edge.wayId);
}
const junctionOsmIds = new Set(osmIds.filter((id) => undirected.get(id).size !== 2 || wayUse.get(id).size > 1));

const nodes = osmIds.map((osmId, id) => {
  const node = nodeByOsmId.get(osmId);
  return { id, osmId, lat: node.lat, lon: node.lon, junction: junctionOsmIds.has(osmId) };
});
const edges = [...bestByPair.values()].map((edge, id) => ({
  id,
  from: internalId.get(edge.from),
  to: internalId.get(edge.to),
  weight: edge.weight,
  distance: edge.distance,
  speed: edge.speed,
  highway: edge.highway,
  name: edge.name,
  wayId: edge.wayId,
  path: [internalId.get(edge.from), internalId.get(edge.to)]
}));

// Build an exact corridor-contracted graph. Only degree-two geometry vertices are
// skipped, and every contracted edge retains the complete base-node path.
const edgeByPair = new Map(edges.map((edge) => [`${edge.from}>${edge.to}`, edge]));
const internalNeighbours = new Map();
for (const [osmId, values] of undirected) {
  internalNeighbours.set(internalId.get(osmId), new Set([...values].map((value) => internalId.get(value))));
}
const junctionIds = nodes.filter((node) => node.junction).map((node) => node.id);
const shortcutByPair = new Map();
for (const start of junctionIds) {
  for (const first of internalNeighbours.get(start) || []) {
    const path = [start, first];
    let previous = start;
    let current = first;
    while (!nodes[current].junction) {
      const next = [...internalNeighbours.get(current)].find((id) => id !== previous);
      if (next === undefined || path.includes(next)) break;
      path.push(next);
      previous = current;
      current = next;
    }
    if (!nodes[current].junction) continue;
    const constituent = [];
    let valid = true;
    for (let index = 0; index + 1 < path.length; index += 1) {
      const edge = edgeByPair.get(`${path[index]}>${path[index + 1]}`);
      if (!edge) {
        valid = false;
        break;
      }
      constituent.push(edge);
    }
    if (!valid || !constituent.length) continue;
    const weight = constituent.reduce((sum, edge) => sum + edge.weight, 0);
    const key = `${start}>${current}`;
    const shortcut = {
      id: 0,
      from: start,
      to: current,
      weight,
      distance: constituent.reduce((sum, edge) => sum + edge.distance, 0),
      speed: constituent.reduce((sum, edge) => sum + edge.speed * edge.distance, 0) /
        constituent.reduce((sum, edge) => sum + edge.distance, 0),
      highway: constituent[0].highway,
      name: constituent[0].name,
      wayId: constituent[0].wayId,
      path
    };
    const previousShortcut = shortcutByPair.get(key);
    if (!previousShortcut || shortcut.weight < previousShortcut.weight) shortcutByPair.set(key, shortcut);
  }
}
const shortcuts = [...shortcutByPair.values()].map((edge, index) => ({ ...edge, id: edges.length + index }));
const maxSpeedKph = Math.max(...edges.map((edge) => edge.speed));
const graph = {
  meta: {
    name: "San Francisco · Downtown & SoMa",
    generatedAt: new Date().toISOString(),
    osmTimestamp: osm.osm3s?.timestamp_osm_base || "unknown",
    attribution: "© OpenStreetMap contributors · ODbL",
    query,
    bounds,
    maxSpeedKph,
    rawWayCount: ways.length
  },
  nodes,
  edges,
  shortcuts
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(graph)}\n`, "utf8");
console.log(`Wrote ${outputPath}`);
console.log(`${nodes.length.toLocaleString()} nodes · ${edges.length.toLocaleString()} directed edges · ${shortcuts.length.toLocaleString()} contracted edges`);
