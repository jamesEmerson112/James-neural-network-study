# Map data

The checked-in road graph in `src/data/san-francisco.graph.json` is derived
from OpenStreetMap data.

© OpenStreetMap contributors. The source data is available under the Open Data
Commons Open Database License (ODbL):

- <https://www.openstreetmap.org/copyright>
- <https://opendatacommons.org/licenses/odbl/>

The exact Overpass query, source timestamp, bounds, and generation timestamp are
embedded in the graph's `meta` object. Run `npm run data:refresh` to regenerate
the extract. Public Overpass servers are shared infrastructure, so refresh the
snapshot deliberately rather than on every application load.
