"""A Typer CLI to incrementally compute embeddings for a directory of images and write them to a SQLite DB.

Example command line:
```sh
python -m crossfilter.inference.compute_embeddings_cli \
    --embedding_type SIGLIP2 \
    --input_dir /path/to/images_by_uuid \
    --output_embeddings_db /path/to/output.db \
    --batch_size 10 \
    --recompute_existing_embeddings=false \
    --reproject_umap_embeddings=true

```


How this works:
* Images are expected to be named as "<uuid>.jpg" in any subdirectory of the input directory.
* There's a generic "EMBEDDINGS" table in the output DB for all embedding types.
  * The table's primary key column is "UUID".
  * There's a second column called "EMBEDDING" with a msgpack_numpy encoded 1D numpy array.
* When the CLI starts, it figures out which "<uuid>.jpg" files are missing embeddings in the table (unless `--recompute_existing_embeddings` is set to `true`).
* Embedding computation loop:
    * For the missing embeddings, it computes them in batches of `--batch-size` using the `compute_image_embeddings` function from `siglip2_embedding_functions.py` (refactored to allow for incremental output to a sink, including a DB write queue).
    * The embeddings are enqueued to be written to the DB as they are computed, ideally in parallel with the computation.
    * A TQDM progress bar is shown for the embedding computation loop.
    * When that finishes, we join the DB write queue and wait for all the embeddings to be written to the DB.
* UMAP computation:
    * Once all embeddings are computed and written to the DB, if `--reproject_umap_embeddings` is set, the CLI will reproject all the available embeddings to the UMAP space.
    * The UMAP model is stored into the DB as BLOB in a table called UMAP_MODEL, with a single column called MODEL.
    * The embeddings are stored in a DB table called {embedding_type}_UMAP_HAVERSINE, with columns UUID (primary key), SEMANTIC_EMBEDDING_UMAP_LATITUDE, SEMANTIC_EMBEDDING_UMAP_LONGITUDE.
    * Nice-to-have: we seed the UMAP embedding with existing embeddings, so that they don't move "too much" when new UUIDs are added.
"""

import logging
import pickle
import threading
from pathlib import Path
from queue import Queue
from uuid import UUID

import msgpack
import msgpack_numpy
import pandas as pd
import typer
from sqlalchemy import (
    Column,
    LargeBinary,
    MetaData,
    Table,
    create_engine,
)
from sqlalchemy.engine import Engine
from tqdm import tqdm

from crossfilter.core.schema import (
    EmbeddingType,
    EmbeddingsTables,
    SchemaColumns,
)
from crossfilter.data_ingestion.sqlite_utils import upsert_dataframe_to_sqlite
from crossfilter.inference.run_umap import run_umap_projection

logger = logging.getLogger(__name__)


def _get_embedder_instance(embedding_type: EmbeddingType):
    """Get the appropriate embedder instance based on embedding type."""
    if embedding_type == EmbeddingType.SIGLIP2:
        from crossfilter.inference.siglip2_embedding_functions import (
            SigLIP2Embedder,
        )

        return SigLIP2Embedder()
    elif embedding_type == EmbeddingType.FAKE_EMBEDDING_FOR_TESTING:
        from crossfilter.inference.fake_embedding_functions import (
            FakeEmbedder,
        )

        return FakeEmbedder()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")


app = typer.Typer(
    help="Compute embeddings for images and store them in SQLite database"
)

# Enable msgpack_numpy for serialization/deserialization
msgpack_numpy.patch()


def _create_database_schema(
    engine: Engine, embedding_type: EmbeddingType
) -> Table:
    """Create database schema and return UMAP model table object."""
    metadata = MetaData()

    # Create UMAP model table
    umap_model_table = Table(
        EmbeddingsTables.UMAP_MODEL,
        metadata,
        Column("MODEL", LargeBinary, nullable=False),
    )

    # Note: Both embeddings table and UMAP haversine table are created automatically by upsert_dataframe_to_sqlite

    # Create all tables
    metadata.create_all(engine)

    return umap_model_table


def _scan_image_files(input_dir: Path) -> dict[str, Path]:
    """Scan input directory for UUID.jpg files and return mapping of UUID -> Path."""
    logger.info(f"Scanning for <uuid>.jpg files in {input_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    uuid_to_path: dict[str, Path] = {}

    # Find all .jpg files recursively
    for jpg_path in input_dir.rglob("*.jpg"):
        # Extract UUID from filename (remove .jpg extension)
        uuid_str = jpg_path.stem
        if uuid_str in uuid_to_path:
            logger.warning(f"Duplicate UUID found in {input_dir}: {uuid_str=}")
            continue
        uuid_to_path[uuid_str] = jpg_path

    logger.info(f"Found {len(uuid_to_path)=} .jpg files")
    return uuid_to_path


def _get_existing_embeddings(engine: Engine) -> set[str]:
    """Get set of UUIDs that already have embeddings in the database."""
    try:
        query = f"SELECT {SchemaColumns.UUID_STRING} FROM {EmbeddingsTables.IMAGE_EMBEDDINGS}"
        df = pd.read_sql(query, engine)
        return set(df[SchemaColumns.UUID_STRING].values)
    except Exception:
        # Table doesn't exist yet, return empty set
        return set()


def _write_embeddings_worker(
    write_queue: Queue,
    output_embeddings_db: Path,
    stop_event: threading.Event,
) -> None:
    """Worker thread that writes embeddings to database using upsert_dataframe_to_sqlite."""

    batch_items = []
    
    while not stop_event.is_set() or not write_queue.empty():
        # Collect batch of items from queue
        while len(batch_items) < 100:  # Batch size
            try:
                item = write_queue.get(timeout=0.1)
                if item is None:  # Sentinel value to stop
                    break
                batch_items.append(item)
                write_queue.task_done()
            except Exception:
                # Timeout or queue empty - process what we have
                break
        
        # Process batch if we have items
        if batch_items:
            try:
                # Create DataFrame from batch items
                uuids = []
                embeddings = []
                
                for uuid_str, embedding in batch_items:
                    uuids.append(uuid_str)
                    embeddings.append(embedding)
                
                batch_df = pd.DataFrame({
                    SchemaColumns.UUID_STRING: uuids,
                    SchemaColumns.SEMANTIC_EMBEDDING: embeddings,
                })
                
                # Use the utility function for upsert
                upsert_dataframe_to_sqlite(
                    batch_df, 
                    output_embeddings_db, 
                    EmbeddingsTables.IMAGE_EMBEDDINGS
                )
                
                logger.debug(f"Wrote batch of {len(batch_items)} embeddings to database")
                batch_items.clear()
                
            except Exception as e:
                logger.error(f"Failed to write batch to database: {e}")
                # Re-raise the exception to avoid silent failures
                # The main thread will handle this appropriately
                raise
        
        # Check for sentinel or stop condition
        if not write_queue.empty():
            continue
        elif stop_event.is_set():
            break


def _compute_embeddings_batch(
    batch_paths: list[Path],
    batch_uuids: list[str],
    write_queue: Queue,
    embedder,
) -> None:
    """Compute embeddings for a batch of images and enqueue them for writing."""
    try:
        # Compute embeddings for the batch using the embedder instance
        embeddings = embedder.compute_image_embeddings(batch_paths)

        # Enqueue each embedding for writing
        for uuid_str, embedding in zip(batch_uuids, embeddings):
            write_queue.put((uuid_str, embedding))

    except Exception as e:
        logger.error(f"Error computing embeddings for batch: {e}")
        raise


def _load_embeddings_from_db(engine: Engine) -> pd.DataFrame:
    """Load all embeddings from database into a DataFrame."""
    
    try:
        query = f"""
        SELECT {SchemaColumns.UUID_STRING}, {SchemaColumns.SEMANTIC_EMBEDDING}
        FROM {EmbeddingsTables.IMAGE_EMBEDDINGS}
        """
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return pd.DataFrame(
                columns=[SchemaColumns.UUID_STRING, SchemaColumns.SEMANTIC_EMBEDDING]
            )
        
        # Deserialize embeddings using map operation
        df[SchemaColumns.SEMANTIC_EMBEDDING] = df[SchemaColumns.SEMANTIC_EMBEDDING].map(
            lambda blob: msgpack.unpackb(blob)
        )
        
        return df
    except Exception:
        # Table doesn't exist yet, return empty DataFrame
        return pd.DataFrame(
            columns=[SchemaColumns.UUID_STRING, SchemaColumns.SEMANTIC_EMBEDDING]
        )


def _store_umap_model(engine, umap_model_table: Table, umap_model) -> None:
    """Store UMAP model in database."""

    # Serialize model with pickle
    model_blob = pickle.dumps(umap_model)

    with engine.connect() as conn:
        # Clear existing model first
        conn.execute(umap_model_table.delete())

        # Insert new model
        conn.execute(umap_model_table.insert().values(MODEL=model_blob))
        conn.commit()


def _upsert_umap_embeddings(output_embeddings_db: Path, embedding_type: EmbeddingType, df: pd.DataFrame) -> None:
    """Upsert UMAP embeddings into the haversine table."""
    
    # Filter to rows that have valid UMAP coordinates
    valid_umap_mask = (
        df[SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LATITUDE].notna() &
        df[SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LONGITUDE].notna()
    )
    
    if not valid_umap_mask.any():
        logger.warning("No valid UMAP coordinates to upsert")
        return
    
    # Select only the columns we need for the UMAP haversine table
    umap_df = df[valid_umap_mask][[
        SchemaColumns.UUID_STRING,
        SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LATITUDE,
        SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LONGITUDE,
    ]].copy()
    
    # Use the existing utility function for upsert
    table_name = f"{embedding_type}_UMAP_HAVERSINE"
    upsert_dataframe_to_sqlite(umap_df, output_embeddings_db, table_name)
    
    logger.info(f"Upserted {len(umap_df)} UMAP embeddings into {table_name} table")


@app.command()
def main(
    embedding_type: EmbeddingType = typer.Option(
        EmbeddingType.SIGLIP2,
        "--embedding_type",
        help="Type of embedding to compute (SIGLIP2 or FAKE_EMBEDDING_FOR_TESTING)",
    ),
    input_dir: Path = typer.Option(
        ..., "--input_dir", help="Directory containing images named as <uuid>.jpg"
    ),
    output_embeddings_db: Path = typer.Option(
        ...,
        "--output_embeddings_db",
        help="Path to SQLite database for storing embeddings",
    ),
    batch_size: int = typer.Option(
        10, "--batch_size", help="Batch size for embedding computation"
    ),
    recompute_existing_embeddings: bool = typer.Option(
        False,
        "--recompute_existing_embeddings/--no-recompute_existing_embeddings",
        help="Whether to recompute embeddings that already exist in the database",
    ),
    reproject_umap_embeddings: bool = typer.Option(
        True,
        "--reproject_umap_embeddings/--no-reproject_umap_embeddings",
        help="Whether to reproject embeddings to UMAP space after computation",
    ),
) -> None:
    """Compute embeddings for images and store them in SQLite database."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting embedding computation with parameters:")
    logger.info(f"  embedding_type: {embedding_type}")
    logger.info(f"  input_dir: {input_dir}")
    logger.info(f"  output_embeddings_db: {output_embeddings_db}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  recompute_existing_embeddings: {recompute_existing_embeddings}")
    logger.info(f"  reproject_umap_embeddings: {reproject_umap_embeddings}")

    # Validate input directory
    if not input_dir.exists():
        raise typer.BadParameter(f"Input directory does not exist: {input_dir}")

    # Create output directory if needed
    output_embeddings_db.parent.mkdir(parents=True, exist_ok=True)

    # Create database engine
    engine = create_engine(f"sqlite:///{output_embeddings_db}")

    # Create database schema
    umap_model_table = _create_database_schema(engine, embedding_type)

    # Scan for image files
    uuid_to_path = _scan_image_files(input_dir)

    if not uuid_to_path:
        logger.warning("No valid UUID.jpg files found in input directory")
        return

    # Determine which embeddings need to be computed
    if recompute_existing_embeddings:
        missing_uuids = set(uuid_to_path.keys())
        logger.info(f"Recomputing all {len(missing_uuids)} embeddings")
    else:
        existing_uuids = _get_existing_embeddings(engine)
        missing_uuids = set(uuid_to_path.keys()) - existing_uuids
        logger.info(f"Found {len(existing_uuids)} existing embeddings")
        logger.info(f"Need to compute {len(missing_uuids)} new embeddings")

    if not missing_uuids:
        logger.info("No embeddings need to be computed")
    else:
        # Create embedder instance once (efficient model loading)
        logger.info(f"Initializing embedder for {embedding_type}")
        embedder = _get_embedder_instance(embedding_type)

        # Prepare batches for computation
        missing_uuids_list = list(missing_uuids)
        batches = []

        for i in range(0, len(missing_uuids_list), batch_size):
            batch_uuids = missing_uuids_list[i : i + batch_size]
            batch_paths = [uuid_to_path[uuid_str] for uuid_str in batch_uuids]
            batches.append((batch_paths, batch_uuids))

        # Set up database write queue and worker
        write_queue = Queue()
        stop_event = threading.Event()

        # Start database writer thread
        writer_thread = threading.Thread(
            target=_write_embeddings_worker,
            args=(write_queue, output_embeddings_db, stop_event),
        )
        writer_thread.start()

        try:
            # Compute embeddings in batches with progress bar
            logger.info(f"Computing embeddings in {len(batches)} batches")

            for batch_paths, batch_uuids in tqdm(batches, desc="Computing embeddings"):
                _compute_embeddings_batch(
                    batch_paths, batch_uuids, write_queue, embedder
                )

            # Wait for all writes to complete
            logger.info("Waiting for all embeddings to be written to database...")
            write_queue.join()

        finally:
            # Wait for all items to be processed first
            write_queue.join()
            # Then signal to stop and wait for thread to finish processing remaining items
            write_queue.put(None)  # Sentinel value
            stop_event.set()
            writer_thread.join()

    # Run UMAP projection if requested
    if reproject_umap_embeddings:
        logger.info("Loading embeddings for UMAP projection...")
        df = _load_embeddings_from_db(engine)

        if len(df) < 20:
            logger.warning("Two few embeddings found for UMAP projection.")
        else:
            logger.info(f"Running UMAP projection on {len(df)} embeddings...")

            # Run UMAP projection
            umap_model = run_umap_projection(
                df,
                embedding_column=SchemaColumns.SEMANTIC_EMBEDDING,
                output_lat_column=SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LATITUDE,
                output_lon_column=SchemaColumns.SEMANTIC_EMBEDDING_UMAP_LONGITUDE,
            )
            
            # Upsert the UMAP embeddings for the images into the haversine table
            _upsert_umap_embeddings(output_embeddings_db, embedding_type, df)

            # Store UMAP model in database
            _store_umap_model(engine, umap_model_table, umap_model)
            logger.info("UMAP model stored in database")

    logger.info("Embedding computation completed successfully")


if __name__ == "__main__":
    app()
