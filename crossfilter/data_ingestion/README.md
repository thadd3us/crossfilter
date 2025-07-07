There are several different CLI programs here than can import data and upsert it into a sqlite database table
with a schema based on //crossfilter/core/schema.py.

# Sample commands

## Ingest GPX file data
```sh
uv run python crossfilter/data_ingestion/gpx/ingest_gpx_files.py ~/Downloads ~/crossfilter/data.sqlite
```

## Ingest Adobe Lightroom data
```sh
# Work laptop:
uv run python crossfilter/data_ingestion/lightroom/ingest_lightroom_catalogs.py \
    ~/personal/non_red ~/crossfilter/data.sqlite \
    --sqlite_db_with_clip_embeddings ~/personal/lightroom_embedding_vectors.sqlite \
    --output_umap_transformation_file ~/crossfilter/clip_umap_transformation.pickle

# Personal laptop
uv run python crossfilter/data_ingestion/lightroom/ingest_lightroom_catalogs.py \
     ~/datasets/lightroom_non_red/ ~/crossfilter/data.sqlite \
    --sqlite_db_with_clip_embeddings ~/monorepo/data/thad/embeddings/images.sqlite \
    --output_umap_transformation_file ~/crossfilter/clip_umap_transformation.pickle

```
