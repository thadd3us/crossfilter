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

### Options

- `--embedding_type`: Type of embedding to compute (default: SIGLIP2)
- `--input_dir`: Directory containing images named as `<uuid>.jpg`
- `--output_embeddings_db`: Path to SQLite database for storing embeddings
- `--batch_size`: Batch size for embedding computation (default: 10)
- `--recompute_existing_embeddings`: Whether to recompute embeddings that already exist
- `--no-recompute_existing_embeddings`: Don't recompute existing embeddings (default)
- `--reproject_umap_embeddings`: Whether to reproject embeddings to UMAP space (default: true)
- `--no-reproject_umap_embeddings`: Don't reproject embeddings to UMAP space

### Input Format

Images should be named as `<uuid>.jpg` and can be in any subdirectory of the input directory. The UUID must be a valid UUID format.

### Output Database Schema

The CLI creates the following tables:

#### `{EMBEDDING_TYPE}_EMBEDDINGS`
- `UUID` (TEXT PRIMARY KEY): The UUID of the image
- `EMBEDDING` (BLOB): The embedding vector serialized with msgpack

#### `{EMBEDDING_TYPE}_UMAP_MODEL`
- `MODEL` (BLOB): The UMAP model serialized with pickle

### Example

```bash
# Compute embeddings for all images in a directory
python -m crossfilter.inference.compute_embeddings_cli \
    --input_dir ./images \
    --output_embeddings_db ./embeddings.db \
    --batch_size 8

# Recompute all embeddings and skip UMAP projection
python -m crossfilter.inference.compute_embeddings_cli \
    --input_dir ./images \
    --output_embeddings_db ./embeddings.db \
    --batch_size 8 \
    --recompute_existing_embeddings \
    --no-reproject_umap_embeddings
```

### Architecture

The CLI follows these steps:

1. **Scan Input Directory**: Recursively finds all `*.jpg` files with valid UUID names
2. **Check Existing Embeddings**: Queries the database to find which embeddings already exist
3. **Compute Missing Embeddings**: Processes images in batches using the specified embedding model
4. **Parallel Database Writes**: Uses a background worker thread to write embeddings as they're computed
5. **UMAP Projection**: If enabled, loads all embeddings and projects them to 2D space
6. **Store UMAP Model**: Saves the UMAP model to the database for future use

### Performance

The CLI is designed to handle large datasets efficiently:

- **Parallel Processing**: Computation and database writes happen in parallel
- **Batch Processing**: Configurable batch sizes balance memory usage and throughput
- **Incremental Updates**: Only processes new images, allowing for efficient re-runs
- **Progress Tracking**: Shows progress for long-running operations

### Error Handling

The CLI includes comprehensive error handling:

- **File Validation**: Checks that input directories exist and contain valid images
- **Database Validation**: Creates tables if they don't exist
- **UUID Validation**: Only processes files with valid UUID names
- **Graceful Shutdown**: Properly cleans up threads and database connections

## Other Files

- `siglip2_embedding_functions.py`: Core embedding computation functions
- `run_umap.py`: UMAP projection functions with spherical geometry
- `*_test.py`: Comprehensive test suites for all modules