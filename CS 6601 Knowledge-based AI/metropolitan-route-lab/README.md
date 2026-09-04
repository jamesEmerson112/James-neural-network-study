# Streetwise: a real-map search laboratory

This frontend compares exact shortest-path algorithms on a reproducible,
directed OpenStreetMap road graph from downtown San Francisco. Unlike the companion
single-file lesson, both the street geometry and routing topology are real.

## Run it

```powershell
npm install
npm run dev
```

Validation:

```powershell
npm test
npm run build
```

Refresh the bounded OSM snapshot only when needed:

```powershell
npm run data:refresh
```

## Architecture

- React, TypeScript, Vite, and MapLibre GL JS
- an immutable OSM graph snapshot for reproducible comparisons
- a Web Worker for graph preprocessing and route queries
- IndexedDB for the landmark and exact-reach index
- Dijkstra as the correctness oracle for every optimized method

The graph respects OSM one-way tags, roundabouts, access tags, road classes,
and tagged/fallback speeds. It does **not** yet model turn-restriction relations,
live traffic, signals, or lane-level costs, so it is an algorithms laboratory—not
a navigation product.

“Shortcuts” here are exact degree-two corridor contractions. They preserve and
unpack the original road-node path. The 2006 *Reach for A\** implementation used
more sophisticated shortcut-aware reach preprocessing at continental scale;
the UI calls out that distinction.

## Basemap policy

For ordinary local interactive use, the fallback style loads standard OSM
raster tiles with visible attribution. Set `VITE_MAP_STYLE_URL` to use a
MapLibre-compatible provider or self-hosted style. Do not bulk-download,
prefetch, or use the community OSM tile service as an offline/production tile
backend. See <https://operations.osmfoundation.org/policies/tiles/>.
