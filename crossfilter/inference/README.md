# Crossfilter Inference Tools

This directory contains tools for computing embeddings and running inference on images using various models.

## compute_embeddings_cli.py

A command-line interface for computing embeddings for images and storing them in a SQLite database. The CLI supports incremental computation, UMAP projection, and parallel processing.

### Features

- **Incremental Processing**: Only computes embeddings for images that don't already exist in the database
- **Batch Processing**: Processes images in configurable batches for efficient memory usage
- **Parallel Database Writes**: Uses a background worker thread to write embeddings to the database in parallel with computation
- **UMAP Projection**: Optionally projects embeddings to 2D space using UMAP with spherical geometry
- **Progress Tracking**: Shows progress bars for long-running operations
- **SQLAlchemy Integration**: Uses SQLAlchemy for robust database operations

### Usage

```bash
python -m crossfilter.inference.compute_embeddings_cli \
    --embedding_type SIGLIP2 \
    --input_dir /path/to/images_by_uuid \
    --output_embeddings_db /path/to/output.db \
    --batch_size 10 \
    --recompute_existing_embeddings \
    --reproject_umap_embeddings
```

### Output Database Schema

The CLI creates the following tables:

#### `{EMBEDDING_TYPE}_EMBEDDINGS`
- `UUID` (TEXT PRIMARY KEY): The UUID of the image
- `EMBEDDING` (BLOB): The embedding vector serialized with msgpack

#### `{EMBEDDING_TYPE}_UMAP_MODEL`
- `MODEL` (BLOB): The UMAP model serialized with pickle

