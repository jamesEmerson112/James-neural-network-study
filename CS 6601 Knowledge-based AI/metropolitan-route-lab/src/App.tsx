import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import graphSnapshot from "./data/san-francisco.graph.json";
import { MapView } from "./components/MapView";
import { deleteIndex, readIndex, writeIndex } from "./data/indexCache";
import type {
  AlgorithmId,
  PreprocessData,
  RoadGraphData,
  SearchResult,
  WorkerResponse
} from "./types";

const graph = graphSnapshot as RoadGraphData;

const ALGORITHMS: Record<AlgorithmId, {
  title: string;
  eyebrow: string;
  formula: string;
  description: string;
  year: string;
}> = {
  dijkstra: {
    title: "Dijkstra",
    eyebrow: "Uniform expansion",
    formula: "priority(v) = g(v)",
    description: "Settles the cheapest known arrival next. It is the exact oracle every other result must match.",
    year: "1959"
  },
  bidijkstra: {
    title: "Bidirectional Dijkstra",
    eyebrow: "Two meeting waves",
    formula: "min g→ + min g← ≥ best complete route",
    description: "Runs one exact search from each endpoint and stops when neither frontier can improve the best meeting route.",
    year: "1960s"
  },
  astar: {
    title: "A* · straight line",
    eyebrow: "Geometric guidance",
    formula: "priority(v) = g(v) + hstraight(v, T)",
    description: "Adds an optimistic travel-time estimate. It knows the destination's direction but not rivers or one-way detours.",
    year: "1968"
  },
  alt: {
    title: "ALT",
    eyebrow: "A* + Landmarks + Triangle inequality",
    formula: "h(v,T) = maxL { d(L,T) − d(L,v), d(v,L) − d(T,L) }",
    description: "Precomputed landmark distances produce a road-aware lower bound. The route is measured with landmarks; it need not visit them.",
    year: "2005"
  },
  reach: {
    title: "Reach",
    eyebrow: "Safe hierarchy pruning",
    formula: "prune v when reach(v) < min(g(v), lowerBound(v,T))",
    description: "A low-reach local vertex cannot sit in the middle of a long shortest route, so it can be rejected safely.",
    year: "2004"
  },
  reachshort: {
    title: "Reach + corridor shortcuts",
    eyebrow: "Prune + compress",
    formula: "reach pruning on an exact degree-two contraction",
    description: "The query skips geometry-only road points, while every contracted edge retains the original nodes needed to draw the route.",
    year: "Teaching analogue"
  },
  combo: {
    title: "Reach + Shortcuts + ALT",
    eyebrow: "Aim + prune + jump",
    formula: "priority = g + hALT; prune by reach; traverse contracted corridors",
    description: "Landmarks aim the queue, reach rejects irrelevant local detail, and exact corridor contractions reduce graph steps.",
    year: "2006 idea"
  }
};

const ORDER = Object.keys(ALGORITHMS) as AlgorithmId[];
const CACHE_KEY = `san-francisco:${graph.meta.osmTimestamp}:reach-v2`;

const PRESETS = [
  { name: "Ferry Building → City Hall", source: [37.7952, -122.3937], target: [37.7794, -122.4192] },
  { name: "Civic Center → Salesforce Transit Center", source: [37.7793, -122.4192], target: [37.7897, -122.3960] },
  { name: "SoMa → Chinatown", source: [37.7765, -122.4100], target: [37.7940, -122.4078] }
] as const;

function nearestJunction(lat: number, lon: number): number {
  let selected = graph.nodes.find((node) => node.junction)?.id ?? 0;
  let best = Number.POSITIVE_INFINITY;
  for (const node of graph.nodes) {
    if (!node.junction) continue;
    const dx = (node.lon - lon) * Math.cos(lat * Math.PI / 180);
    const dy = node.lat - lat;
    const distance = dx * dx + dy * dy;
    if (distance < best) {
      best = distance;
      selected = node.id;
    }
  }
  return selected;
}

function formatMs(value: number): string {
  if (value < 0.01) return "<0.01 ms";
  if (value < 10) return `${value.toFixed(2)} ms`;
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

function formatDuration(minutes: number): string {
  if (!Number.isFinite(minutes)) return "No route";
  return minutes < 1 ? `${Math.round(minutes * 60)} sec` : `${minutes.toFixed(2)} min`;
}

function formatDistance(meters: number): string {
  const miles = meters / 1609.344;
  return miles < 0.1 ? `${Math.round(meters)} m` : `${miles.toFixed(2)} mi`;
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong>{hint && <small>{hint}</small>}</div>;
}

export function App() {
  const initial = PRESETS[0];
  const [source, setSource] = useState(() => nearestJunction(initial.source[0], initial.source[1]));
  const [target, setTarget] = useState(() => nearestJunction(initial.target[0], initial.target[1]));
  const [placement, setPlacement] = useState<"source" | "target">("source");
  const [algorithm, setAlgorithm] = useState<AlgorithmId>("alt");
  const [indexes, setIndexes] = useState<PreprocessData>();
  const [progress, setProgress] = useState({ phase: "Loading road graph", current: 0, total: 1 });
  const [workerError, setWorkerError] = useState("");
  const [result, setResult] = useState<SearchResult>();
  const [race, setRace] = useState<SearchResult[]>([]);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(120);
  const [showReach, setShowReach] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [raceBusy, setRaceBusy] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);

  const run = useCallback((selected = algorithm, from = source, to = target) => {
    if (!indexes || !workerRef.current || from === to) return;
    requestId.current += 1;
    setResult(undefined);
    setCursor(0);
    setPlaying(false);
    workerRef.current.postMessage({ type: "search", requestId: requestId.current, algorithm: selected, source: from, target: to });
  }, [algorithm, indexes, source, target]);

  useEffect(() => {
    const worker = new Worker(new URL("./workers/search.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.type === "progress") {
        setProgress({ phase: message.phase, current: message.current, total: message.total });
      } else if (message.type === "ready") {
        setIndexes(message.data);
        setProgress({ phase: message.data.fromCache ? "Index restored from IndexedDB" : "Index ready", current: 1, total: 1 });
        if (!message.data.fromCache) void writeIndex(CACHE_KEY, message.data).catch(() => undefined);
      } else if (message.type === "result") {
        if (message.result.requestId === requestId.current) setResult(message.result);
      } else if (message.type === "race-result") {
        if (message.requestId === requestId.current) {
          setRace(message.results);
          setRaceBusy(false);
        }
      } else if (message.type === "error") {
        setWorkerError(message.message);
        setRaceBusy(false);
      }
    };
    void readIndex(CACHE_KEY)
      .catch(() => undefined)
      .then((cached) => worker.postMessage({ type: "initialize", graph, cached }));
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (indexes) run();
    // Run once when preprocessing completes; later changes call run explicitly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indexes]);

  useEffect(() => {
    if (!result) return;
    setCursor(0);
    setPlaying(true);
  }, [result]);

  useEffect(() => {
    if (!playing || !result) return;
    const timer = window.setInterval(() => {
      setCursor((value) => {
        const next = Math.min(result.events.length, value + Math.max(1, Math.round(speed / 20)));
        if (next === result.events.length) setPlaying(false);
        return next;
      });
    }, 50);
    return () => window.clearInterval(timer);
  }, [playing, result, speed]);

  const selectAlgorithm = (next: AlgorithmId) => {
    setAlgorithm(next);
    setShowReach(next === "reach" || next === "reachshort" || next === "combo");
    setShowShortcuts(next === "reachshort" || next === "combo");
    run(next);
  };

  const pickNode = (node: number) => {
    if (placement === "source") {
      setSource(node);
      setPlacement("target");
      run(algorithm, node, target);
    } else {
      setTarget(node);
      setPlacement("source");
      run(algorithm, source, node);
    }
    setRace([]);
  };

  const applyPreset = (index: number) => {
    const preset = PRESETS[index];
    const nextSource = nearestJunction(preset.source[0], preset.source[1]);
    const nextTarget = nearestJunction(preset.target[0], preset.target[1]);
    setSource(nextSource);
    setTarget(nextTarget);
    setRace([]);
    run(algorithm, nextSource, nextTarget);
  };

  const startRace = () => {
    if (!indexes || !workerRef.current) return;
    requestId.current += 1;
    setRaceBusy(true);
    workerRef.current.postMessage({ type: "race", requestId: requestId.current, source, target });
  };

  const rebuild = async () => {
    if (!workerRef.current) return;
    setIndexes(undefined);
    setResult(undefined);
    setRace([]);
    setProgress({ phase: "Rebuilding indexes", current: 0, total: 1 });
    await deleteIndex(CACHE_KEY).catch(() => undefined);
    workerRef.current.postMessage({ type: "rebuild" });
  };

  const currentEvent = result && cursor > 0 ? result.events[Math.min(cursor, result.events.length) - 1] : undefined;
  const completion = result?.events.length ? cursor / result.events.length : 0;
  const bestExpanded = race.length ? Math.min(...race.map((item) => item.expanded)) : 0;
  const selectedInfo = ALGORITHMS[algorithm];
  const cacheLabel = indexes?.fromCache ? "restored" : "measured now";
  const preprocessingPercent = progress.total ? Math.round(progress.current / progress.total * 100) : 0;

  const sourceNode = graph.nodes[source];
  const targetNode = graph.nodes[target];
  const snapshotDate = useMemo(() => {
    const parsed = new Date(graph.meta.osmTimestamp);
    return Number.isNaN(parsed.valueOf()) ? graph.meta.osmTimestamp : parsed.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }, []);

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Streetwise home">
          <span className="brand-mark">S</span>
          <span>STREETWISE<small>REAL-MAP SEARCH LAB</small></span>
        </a>
        <nav aria-label="Page sections">
          <a href="#laboratory">Laboratory</a>
          <a href="#race">Comparison</a>
          <a href="#lore">Lore</a>
        </nav>
        <div className="data-chip"><i /> OSM snapshot · {snapshotDate}</div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="overline">Knowledge-based AI · Route planning</p>
          <h1>The streets are real.<br /><em>The search is visible.</em></h1>
          <p className="hero-lede">Route through downtown San Francisco on a directed OpenStreetMap graph. Watch seven exact methods solve the same trip—and account for every scan, byte, and millisecond.</p>
          <div className="hero-facts">
            <div><strong>{graph.nodes.length.toLocaleString()}</strong><span>OSM road nodes</span></div>
            <div><strong>{graph.edges.length.toLocaleString()}</strong><span>directed arcs</span></div>
            <div><strong>{graph.shortcuts.length.toLocaleString()}</strong><span>contracted arcs</span></div>
          </div>
        </div>
        <div className="hero-note">
          <span>What changed?</span>
          <p>The basemap is not the graph. Map tiles supply geographic context; the checked-in OSM extract supplies the actual vertices, directed roads, travel costs, and one-way rules searched below.</p>
        </div>
      </section>

      <section className="preprocess-strip" aria-live="polite">
        <div className="preprocess-label"><span className={indexes ? "status-dot ready" : "status-dot"} />{progress.phase}</div>
        <div className="preprocess-track"><i style={{ width: `${preprocessingPercent}%` }} /></div>
        <strong>{preprocessingPercent}%</strong>
        <button type="button" onClick={rebuild} disabled={!indexes}>Rebuild &amp; measure</button>
      </section>

      {workerError && <div className="error-banner" role="alert"><strong>Worker error:</strong> {workerError}</div>}

      <section className="lab-section" id="laboratory">
        <div className="section-heading">
          <div><p className="overline">01 · Interactive laboratory</p><h2>One city. One route. Different knowledge.</h2></div>
          <p>Click the map to move the active endpoint. Every optimized answer is compared with an ordinary Dijkstra run before it receives a green “exact” badge.</p>
        </div>

        <div className="lab-frame">
          <div className="map-column">
            <MapView
              graph={graph}
              source={source}
              target={target}
              result={result}
              cursor={cursor}
              indexes={indexes}
              showReach={showReach}
              showShortcuts={showShortcuts}
              placement={placement}
              onPick={pickNode}
            />
            <div className="map-legend">
              <span><i className="forward" />forward scan</span>
              <span><i className="backward" />backward scan</span>
              <span><i className="pruned" />reach-pruned</span>
              <span><i className="route" />final route</span>
              <span><i className="landmark" />landmark</span>
            </div>
          </div>

          <aside className="control-panel">
            <div className="control-block">
              <label htmlFor="preset">Route</label>
              <select id="preset" defaultValue="0" onChange={(event) => applyPreset(Number(event.target.value))}>
                {PRESETS.map((preset, index) => <option value={index} key={preset.name}>{preset.name}</option>)}
              </select>
              <div className="endpoint-toggle" role="group" aria-label="Endpoint to place">
                <button className={placement === "source" ? "active" : ""} type="button" onClick={() => setPlacement("source")}><b>S</b> Start</button>
                <button className={placement === "target" ? "active" : ""} type="button" onClick={() => setPlacement("target")}><b>T</b> Destination</button>
              </div>
              <div className="coordinates"><span>{sourceNode.lat.toFixed(4)}, {sourceNode.lon.toFixed(4)}</span><span>{targetNode.lat.toFixed(4)}, {targetNode.lon.toFixed(4)}</span></div>
            </div>

            <div className="control-block">
              <label htmlFor="algorithm">Search engine</label>
              <select id="algorithm" value={algorithm} onChange={(event) => selectAlgorithm(event.target.value as AlgorithmId)} disabled={!indexes}>
                {ORDER.map((id) => <option value={id} key={id}>{ALGORITHMS[id].title}</option>)}
              </select>
            </div>

            <article className="algorithm-card">
              <span>{selectedInfo.eyebrow}</span>
              <h3>{selectedInfo.title}</h3>
              <code>{selectedInfo.formula}</code>
              <p>{selectedInfo.description}</p>
            </article>

            <div className="layer-toggles">
              <label><input type="checkbox" checked={showReach} onChange={(event) => setShowReach(event.target.checked)} /> Reach heat</label>
              <label><input type="checkbox" checked={showShortcuts} onChange={(event) => setShowShortcuts(event.target.checked)} /> Shortcut overlay</label>
            </div>

            <div className="playback">
              <button className="play-button" type="button" onClick={() => {
                if (result && cursor >= result.events.length) setCursor(0);
                setPlaying((value) => !value);
              }} disabled={!result} aria-label={playing ? "Pause animation" : "Play animation"}>{playing ? "Ⅱ" : "▶"}</button>
              <button type="button" onClick={() => { setPlaying(false); setCursor((value) => Math.min(result?.events.length ?? 0, value + 1)); }} disabled={!result}>Step</button>
              <button type="button" onClick={() => { setPlaying(false); setCursor(0); }} disabled={!result}>Reset</button>
              <button type="button" onClick={() => run()} disabled={!indexes}>Run again</button>
            </div>
            <label className="timeline-label" htmlFor="timeline"><span>Search timeline</span><b>{cursor.toLocaleString()} / {(result?.events.length ?? 0).toLocaleString()}</b></label>
            <input id="timeline" type="range" min="0" max={result?.events.length ?? 0} value={cursor} onChange={(event) => { setPlaying(false); setCursor(Number(event.target.value)); }} />
            <label className="speed-label" htmlFor="speed"><span>Animation speed</span><b>{speed} scans/s</b></label>
            <input id="speed" type="range" min="20" max="600" step="20" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />

            <div className="event-readout">
              <span style={{ width: `${completion * 100}%` }} />
              <p>{!result ? "Waiting for preprocessing…" : !currentEvent ? "Ready at the start intersection." : currentEvent.pruned ? `Vertex ${currentEvent.node.toLocaleString()} cannot sit deep enough inside this shortest route; reach prunes it.` : `${currentEvent.side === "forward" ? "Forward" : "Backward"} scan settles vertex ${currentEvent.node.toLocaleString()} at g = ${currentEvent.g.toFixed(3)} min.`}</p>
            </div>
          </aside>
        </div>

        <div className="metrics-grid" aria-live="polite">
          <Metric label="Route cost" value={result ? formatDuration(result.cost) : "—"} hint={result ? formatDistance(result.distance) : "weighted travel time"} />
          <Metric label="Settled vertices" value={result?.expanded.toLocaleString() ?? "—"} hint={`${result?.pruned.toLocaleString() ?? 0} safely pruned`} />
          <Metric label="Relaxed arcs" value={result?.relaxed.toLocaleString() ?? "—"} hint="road edges inspected" />
          <Metric label="Peak frontier" value={result?.frontierPeak.toLocaleString() ?? "—"} hint="queue entries" />
          <Metric label="Worker query" value={result ? formatMs(result.queryMs) : "—"} hint="excludes animation" />
          <Metric label="Query memory" value={result ? formatBytes(result.memoryBytes) : "—"} hint="deterministic estimate" />
          <div className={`metric verification ${result?.verified ? "passed" : ""}`}><span>Correctness</span><strong>{result ? result.verified ? "Exact ✓" : "Mismatch" : "—"}</strong><small>checked against Dijkstra</small></div>
        </div>
      </section>

      <section className="race-section" id="race">
        <div className="section-heading light">
          <div><p className="overline">02 · Fair comparison</p><h2>Race every method on the same trip.</h2></div>
          <button className="race-button" type="button" onClick={startRace} disabled={!indexes || raceBusy}>{raceBusy ? "Running in worker…" : "Run the seven-way race"}</button>
        </div>
        <div className="race-grid">
          {ORDER.map((id) => {
            const item = race.find((candidate) => candidate.algorithm === id);
            const ratio = item && bestExpanded ? bestExpanded / item.expanded : 0;
            return <article className={`race-card ${item?.expanded === bestExpanded ? "winner" : ""}`} key={id}>
              <header><span>{ALGORITHMS[id].year}</span>{item?.verified && <b>exact</b>}</header>
              <h3>{ALGORITHMS[id].title}</h3>
              <div className="race-bar"><i style={{ width: `${Math.max(4, ratio * 100)}%` }} /></div>
              <div className="race-values"><strong>{item ? item.expanded.toLocaleString() : "—"}<small>settled</small></strong><strong>{item ? formatMs(item.queryMs) : "—"}<small>worker time</small></strong></div>
            </article>;
          })}
        </div>
        <p className="comparison-note"><strong>Read scans before milliseconds.</strong> A tiny graph produces noisy sub-millisecond timings; settled vertices reveal the algorithmic work more reliably. Shortcuts here are exact degree-two corridor contractions, not the paper’s full shortcut-aware reach implementation.</p>
      </section>

      <section className="index-section">
        <div className="section-heading">
          <div><p className="overline">03 · Pay once, query often</p><h2>The index has a bill.</h2></div>
          <p>ALT and reach are fast at query time because they first manufacture reusable knowledge about a mostly static road network.</p>
        </div>
        <div className="index-ledger">
          <article><span>Graph assembly</span><strong>{indexes ? formatMs(indexes.graphBuildMs) : "—"}</strong><small>adjacency + reverse graph</small></article>
          <article><span>8 landmark runs</span><strong>{indexes ? formatMs(indexes.landmarkMs) : "—"}</strong><small>forward + reverse distances</small></article>
          <article><span>Exact reach</span><strong>{indexes ? formatMs(indexes.reachMs) : "—"}</strong><small>one shortest-path DAG per source</small></article>
          <article><span>Stored index</span><strong>{indexes ? formatBytes(indexes.indexBytes) : "—"}</strong><small>{cacheLabel}</small></article>
        </div>
        <div className="index-explanation">
          <p><b>Why IndexedDB?</b> Reloading should not repeat expensive, deterministic preprocessing. “Rebuild &amp; measure” deliberately clears the cache so you can observe the cost.</p>
          <p><b>What RAM means here:</b> the lab reports allocated numeric structures, not a browser process total polluted by tabs, extensions, rendering, and garbage collection.</p>
        </div>
      </section>

      <section className="lore-section" id="lore">
        <div className="section-heading">
          <div><p className="overline">04 · Definition &amp; lore</p><h2>Route planning learned to reuse geography.</h2></div>
        </div>
        <div className="timeline">
          <article><time>1959</time><h3>Dijkstra</h3><p>Exact distance radiates outward. With nonnegative road costs, the next settled label is final.</p></article>
          <article><time>1968</time><h3>A*</h3><p>Hart, Nilsson, and Raphael add an admissible estimate of work remaining, turning expansion toward the goal.</p></article>
          <article><time>2004</time><h3>Reach</h3><p>Ron Gutman formalizes where a vertex may sit inside shortest paths—a graph-theoretic form of road hierarchy.</p></article>
          <article><time>2005</time><h3>ALT</h3><p>Goldberg and Harrelson use landmarks and triangle inequalities to make A* aware of network obstacles.</p></article>
          <article><time>2006</time><h3>Reach for A*</h3><p>Goldberg, Kaplan, and Werneck combine goal direction, hierarchy pruning, and exact shortcut arcs at continental scale.</p></article>
        </div>
        <p className="source-line">Primary reading: <a href="https://www.microsoft.com/en-us/research/publication/computing-the-shortest-path-a-search-meets-graph-theory-2/" target="_blank" rel="noreferrer">Goldberg &amp; Harrelson, <i>Computing the Shortest Path: A* Search Meets Graph Theory</i> (2005)</a> · <a href="https://doi.org/10.1137/1.9781611972863.13" target="_blank" rel="noreferrer">Goldberg, Kaplan &amp; Werneck, <i>Reach for A*</i> (2006)</a>.</p>

        <div className="historic-table-wrap">
          <div className="table-intro"><span>Historical benchmark</span><h3>North America · 30M vertices · 16 landmarks</h3><p>These supplied lecture-slide figures are historical measurements, not predictions for this browser.</p></div>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Method</th><th>Preprocess</th><th>Index</th><th>Avg. scans</th><th>Max scans</th><th>Query</th></tr></thead>
              <tbody>
                <tr><td>Bidirectional Dijkstra</td><td>—</td><td>0.5 GB</td><td>10,255,356</td><td>27,166,866</td><td>7,633.9 ms</td></tr>
                <tr><td>ALT</td><td>1.6 h</td><td>2.3 GB</td><td>250,381</td><td>3,584,377</td><td>393.4 ms</td></tr>
                <tr><td>Reach</td><td colSpan={5}>preprocessing reported as impractical</td></tr>
                <tr><td>Reach + Short</td><td>11.3 h</td><td>1.8 GB</td><td>14,684</td><td>24,618</td><td>17.4 ms</td></tr>
                <tr><td>Reach + Short + ALT</td><td>12.9 h</td><td>3.6 GB</td><td>1,595</td><td>7,450</td><td>3.7 ms</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <footer>
        <div><strong>STREETWISE</strong><span>{graph.meta.name}</span></div>
        <p>{graph.meta.attribution}. Snapshot query and license are recorded with the project. Educational routing only: turn restrictions and live traffic are not modeled.</p>
        <a href="../06_metropolitan_route_search.html">Open the compact concept lesson ↗</a>
      </footer>
    </main>
  );
}
