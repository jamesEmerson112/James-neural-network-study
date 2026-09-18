import { useEffect, useRef, useState } from "react";
import maplibregl, { type GeoJSONSource, type Map as MapLibreMap, type StyleSpecification } from "maplibre-gl";
import type { FeatureCollection, LineString, Point } from "geojson";
import type { PreprocessData, RoadGraphData, SearchResult } from "../types";
import "maplibre-gl/dist/maplibre-gl.css";

interface Props {
  graph: RoadGraphData;
  source: number;
  target: number;
  result?: SearchResult;
  cursor: number;
  indexes?: PreprocessData;
  showReach: boolean;
  showShortcuts: boolean;
  placement: "source" | "target";
  onPick: (node: number) => void;
}

const emptyLines: FeatureCollection<LineString> = { type: "FeatureCollection", features: [] };
const emptyPoints: FeatureCollection<Point> = { type: "FeatureCollection", features: [] };

function fallbackStyle(): StyleSpecification {
  return {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors"
      }
    },
    layers: [
      {
        id: "osm-basemap",
        type: "raster",
        source: "osm",
        paint: { "raster-saturation": -0.72, "raster-brightness-max": 0.7, "raster-contrast": 0.18 }
      }
    ]
  };
}

function roadFeatures(graph: RoadGraphData): FeatureCollection<LineString> {
  const used = new Set<string>();
  return {
    type: "FeatureCollection",
    features: graph.edges.flatMap((edge) => {
      const key = edge.from < edge.to ? `${edge.from}:${edge.to}` : `${edge.to}:${edge.from}`;
      if (used.has(key)) return [];
      used.add(key);
      const a = graph.nodes[edge.from];
      const b = graph.nodes[edge.to];
      return [{
        type: "Feature" as const,
        properties: { highway: edge.highway, name: edge.name },
        geometry: { type: "LineString" as const, coordinates: [[a.lon, a.lat], [b.lon, b.lat]] }
      }];
    })
  };
}

function shortcutFeatures(graph: RoadGraphData): FeatureCollection<LineString> {
  return {
    type: "FeatureCollection",
    features: graph.shortcuts.filter((edge) => edge.path.length > 2).map((edge) => ({
      type: "Feature",
      properties: { skipped: edge.path.length - 2 },
      geometry: {
        type: "LineString",
        coordinates: edge.path.map((id) => [graph.nodes[id].lon, graph.nodes[id].lat])
      }
    }))
  };
}

function setData(map: MapLibreMap, id: string, data: FeatureCollection): void {
  (map.getSource(id) as GeoJSONSource | undefined)?.setData(data);
}

function addSources(map: MapLibreMap, graph: RoadGraphData): void {
  map.addSource("road-graph", { type: "geojson", data: roadFeatures(graph) });
  map.addSource("shortcut-graph", { type: "geojson", data: shortcutFeatures(graph) });
  map.addSource("reach-nodes", { type: "geojson", data: emptyPoints });
  map.addSource("search-nodes", { type: "geojson", data: emptyPoints });
  map.addSource("landmarks", { type: "geojson", data: emptyPoints });
  map.addSource("final-route", { type: "geojson", data: emptyLines });
  map.addSource("endpoints", { type: "geojson", data: emptyPoints });

  map.addLayer({
    id: "road-graph-casing",
    type: "line",
    source: "road-graph",
    paint: { "line-color": "#061016", "line-opacity": 0.78, "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1.2, 16, 5.2] }
  });
  map.addLayer({
    id: "road-graph-lines",
    type: "line",
    source: "road-graph",
    paint: {
      "line-color": ["match", ["get", "highway"], ["motorway", "trunk", "primary"], "#638b91", ["secondary", "tertiary"], "#52747b", "#38565e"],
      "line-opacity": 0.9,
      "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.7, 16, 2.7]
    }
  });
  map.addLayer({
    id: "reach-heat",
    type: "circle",
    source: "reach-nodes",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 1.5, 16, 4],
      "circle-color": ["interpolate", ["linear"], ["get", "ratio"], 0, "#275c66", 0.35, "#69d7c7", 0.7, "#ffb95c", 1, "#ff6d72"],
      "circle-opacity": 0.72
    }
  });
  map.addLayer({
    id: "shortcut-lines",
    type: "line",
    source: "shortcut-graph",
    paint: { "line-color": "#c2a9ff", "line-width": 2.1, "line-opacity": 0.74, "line-dasharray": [2, 1.5] }
  });
  map.addLayer({
    id: "scanned-nodes",
    type: "circle",
    source: "search-nodes",
    paint: {
      "circle-radius": ["case", ["get", "pruned"], 4.4, 3.3],
      "circle-color": ["case", ["get", "pruned"], "#ff7083", ["==", ["get", "side"], "backward"], "#ff886f", "#58b9ff"],
      "circle-opacity": 0.76,
      "circle-stroke-width": 0.7,
      "circle-stroke-color": "#071217"
    }
  });
  map.addLayer({
    id: "landmark-points",
    type: "circle",
    source: "landmarks",
    paint: { "circle-radius": 6, "circle-color": "#c6b0ff", "circle-stroke-width": 2, "circle-stroke-color": "#ffffff" }
  });
  map.addLayer({
    id: "route-casing",
    type: "line",
    source: "final-route",
    paint: { "line-color": "#071217", "line-width": 9, "line-opacity": 0.92 }
  });
  map.addLayer({
    id: "route-line",
    type: "line",
    source: "final-route",
    paint: { "line-color": "#ffd264", "line-width": 5.3, "line-opacity": 1 }
  });
  map.addLayer({
    id: "endpoint-circles",
    type: "circle",
    source: "endpoints",
    paint: {
      "circle-radius": 9,
      "circle-color": ["match", ["get", "kind"], "source", "#66e0ba", "#ffca62"],
      "circle-stroke-width": 3,
      "circle-stroke-color": "#081318"
    }
  });
}

export function MapView(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const latest = useRef(props);
  const [mapMessage, setMapMessage] = useState("");
  latest.current = props;

  const synchronize = () => {
    const map = mapRef.current;
    const current = latest.current;
    if (!map?.isStyleLoaded() || !map.getSource("road-graph")) return;
    const visibleEvents = current.result?.events.slice(0, current.cursor) ?? [];
    setData(map, "search-nodes", {
      type: "FeatureCollection",
      features: visibleEvents.map((event) => {
        const node = current.graph.nodes[event.node];
        return {
          type: "Feature",
          properties: { side: event.side, pruned: event.pruned },
          geometry: { type: "Point", coordinates: [node.lon, node.lat] }
        };
      })
    });
    const route = current.cursor >= (current.result?.events.length ?? 1) ? current.result?.route ?? [] : [];
    setData(map, "final-route", route.length ? {
      type: "FeatureCollection",
      features: [{
        type: "Feature",
        properties: {},
        geometry: { type: "LineString", coordinates: route.map((id) => [current.graph.nodes[id].lon, current.graph.nodes[id].lat]) }
      }]
    } : emptyLines);
    setData(map, "endpoints", {
      type: "FeatureCollection",
      features: [
        { type: "Feature", properties: { kind: "source" }, geometry: { type: "Point", coordinates: [current.graph.nodes[current.source].lon, current.graph.nodes[current.source].lat] } },
        { type: "Feature", properties: { kind: "target" }, geometry: { type: "Point", coordinates: [current.graph.nodes[current.target].lon, current.graph.nodes[current.target].lat] } }
      ]
    });
    const maxReach = Math.max(...(current.indexes?.reach ?? [1]), 1);
    setData(map, "reach-nodes", current.showReach && current.indexes ? {
      type: "FeatureCollection",
      features: current.graph.nodes.filter((node) => node.junction).map((node) => ({
        type: "Feature",
        properties: { ratio: current.indexes!.reach[node.id] / maxReach },
        geometry: { type: "Point", coordinates: [node.lon, node.lat] }
      }))
    } : emptyPoints);
    setData(map, "landmarks", current.indexes ? {
      type: "FeatureCollection",
      features: current.indexes.landmarkIds.map((id) => ({
        type: "Feature",
        properties: {},
        geometry: { type: "Point", coordinates: [current.graph.nodes[id].lon, current.graph.nodes[id].lat] }
      }))
    } : emptyPoints);
    map.setLayoutProperty("shortcut-lines", "visibility", current.showShortcuts ? "visible" : "none");
    map.setLayoutProperty("reach-heat", "visibility", current.showReach ? "visible" : "none");
  };

  useEffect(() => {
    if (!container.current) return;
    const bounds = props.graph.meta.bounds;
    const style = import.meta.env.VITE_MAP_STYLE_URL || fallbackStyle();
    const map = new maplibregl.Map({
      container: container.current,
      style,
      center: [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
      zoom: 14.3,
      minZoom: 11,
      maxZoom: 19,
      attributionControl: false
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: "imperial" }), "bottom-left");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    map.on("load", () => {
      addSources(map, props.graph);
      synchronize();
    });
    let tileErrors = 0;
    map.on("error", () => {
      tileErrors += 1;
      if (tileErrors === 3) setMapMessage("Basemap tiles are unavailable; the checked-in road graph still works offline.");
    });
    map.on("click", (event) => {
      const current = latest.current;
      let nearest = current.source;
      let best = Number.POSITIVE_INFINITY;
      for (const node of current.graph.nodes) {
        if (!node.junction) continue;
        const dx = (node.lon - event.lngLat.lng) * Math.cos(event.lngLat.lat * Math.PI / 180);
        const dy = node.lat - event.lngLat.lat;
        const distance = dx * dx + dy * dy;
        if (distance < best) {
          best = distance;
          nearest = node.id;
        }
      }
      current.onPick(nearest);
    });
    return () => map.remove();
    // The checked-in graph is immutable for the lifetime of this component.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(synchronize, [props.result, props.cursor, props.indexes, props.showReach, props.showShortcuts, props.source, props.target]);

  return (
    <div className="map-shell">
      <div ref={container} className="map" aria-label="Interactive San Francisco routing map" />
      <div className="map-instruction"><span>{props.placement === "source" ? "S" : "T"}</span> Click a road junction to place the {props.placement === "source" ? "start" : "destination"}</div>
      {mapMessage && <div className="map-message" role="status">{mapMessage}</div>}
    </div>
  );
}
