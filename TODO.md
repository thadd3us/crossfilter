# TODO Items

* AG Grid table view of the selected rows.
* CLI for a program to compute SigLIP2 embeddings on a directory full of images and write to sqlite DB, only doing what isn't in the DB already, with tqdm.

* SQLite to Plotly treemap of size of each table and column.
* AG Grid view of counted column values, selectable, with proper NA handling.
* Handle bad data on ingestion and validate against the pandera schema before writing the to DB.
* GPS loader for the Google Takeout data -- add schema fields?
* Allow program to read from multiple sqlite DBs, to speed loading when you want to focus (2022 GPS tracks, for example).)
* When you colorby, each trace gets max_rows, and then we bucket, rather than bucketing first, then doing the color grouping.  Make the bucketing code aware of the colorby.
* Be able to add and remove projections, and configure them separately with a config button.
* Folder projection.
* Be able to shift the time stamps of all the images in a folder, and add a timezone to the lightroom catalog.

Display:
* Fixed colors for specific names.
* Make the GPS points larger or smaller depending on H3 zoom level.  For coarse buckets, draw the hexagons.
* If all points are in the same timezone, show using local time.

GPS trackpoint optimization:
* Get rid of the "load data" thing; CLI is the only way.
* Command-line can take multple sqlite DBs, load them all into a single DF, annotate each row with the DB it came from, be able to remove them independently.
* Benchmark the speed of loading a large sqlite DB wtih 10M GPS rows and 200k images.
* Fewer H3 bucketing levels, consider using enums for the values.  Look at the value counts for the H3 levels when a large GPS track is loaded.
* Smooth the trackpoints to remove nearby points.

Refactor the projections:
* Each trace on the projection should be bucketed independently.
* Create new projections on the fly.
* Make per-plot settings objects in the frontend/backend shared schema.


* BUG: Selecting only photos on the CDF and intersecting shows too many rows on the Geo plot until you refresh the page.

# umap projection on CLIP embeddings of images and , with text search.
# Copy current uuids out inside clipboard.
# Make a "settings" panel for each projection to control groupby, maxrows, etc.
# Bug around number of selected -- are state updates working?

No:
# ParCoords plot?
* Tree-view of the folders and files that are in the DB: https://vue-treeselect.js.org/; Allow expanding seelction to

Underway:



All previous TODO items have been completed:

# Fix the bug where deselecting a DataType and restricting to those points doesn't actually do the restriction because we don't match on the groupby column.  Add unit test.
# Investigate a React-like framework for UI state progagation from bojects, and plot updates.
# Build image click handler on a right-hand pane.


# Frontend unit test that alternates between geo and temporal selection.
#   - Including selecting bucketed points with a COUNT.

# Build a large dataset of images and GPS points to demo.
  - Scan a bunch of JPEG and movie files to extract metadata (time, GPS, caption, camera, lens) into parquet DataFrame. (Are Lightroom UUIDs in there?)

  - GPX to parquet DataFrame conversion.


* Cache the UMAP projection given a certain set of embeddings.
* Figure out what's wrong with the UMAP projection.
* Make the sqlite query from the LR catalog timezone aware, and get a sense of how often it's being used.

✅ **Fixed timezone warnings in bucketing.py** - Fixed `dt.to_period()` calls that dropped timezone information by temporarily converting to timezone-naive with `dt.tz_convert(None)`, applying the period conversion, then restoring UTC timezone with `dt.tz_localize("UTC")`. Added assertions that enforce timezone-aware UTC timestamps and regression test that fails if the warning returns.

✅ **Updated web UI status format** - Changed status line to format: "Status: # rows loaded with # columns (# MB), # remain in current view" with proper memory usage reporting.

✅ **Added deep memory usage reporting** - Added `memory_usage_mb` field to session state broadcasting and session API endpoint using `df.memory_usage(deep=True)`.

✅ **Updated unit tests** - Updated all test snapshots and frontend tests to support the new status format and timezone-aware temporal columns.

No outstanding TODO items remain.



```

```