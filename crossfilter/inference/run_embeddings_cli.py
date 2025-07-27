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
* The output DB uses tables defined in schema.EmbeddingsTables:
  * IMAGE_EMBEDDINGS table with UUID_STRING and SEMANTIC_EMBEDDING columns.
  * UMAP_MODEL table for storing the trained UMAP model.
  * {embedding_type}_UMAP_HAVERSINE table for UMAP projections.
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
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Any

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
    EmbeddingsTables,
    EmbeddingType,
)
from crossfilter.core.schema import SchemaColumns as C, EmbeddingsTables
from crossfilter.core.serialization import (
    deserialize_numpy_array,
    serialize_numpy_array,
)
from crossfilter.data_ingestion.sqlite_utils import upsert_dataframe_to_sqlite
from crossfilter.inference.embedding_interface import EmbeddingInterface
from crossfilter.inference import umap_haversine

logger = logging.getLogger(__name__)


def _get_embedder_instance(embedding_type: EmbeddingType) -> EmbeddingInterface:
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


# def _create_database_schema(engine: Engine, embedding_type: EmbeddingType) -> Table:
#     """Create database schema and return UMAP model table object."""
#     metadata = MetaData()

#     # Create UMAP model table
#     umap_model_table = Table(
#         EmbeddingsTables.UMAP_MODEL,
#         metadata,
#         Column("MODEL", LargeBinary, nullable=False),
#     )

#     # Note: Both embeddings table and UMAP haversine table are created automatically by upsert_dataframe_to_sqlite

#     # Create all tables
#     metadata.create_all(engine)

#     return umap_model_table


def _scan_image_files(input_dir: Path) -> pd.DataFrame:
    """Scan input directory for UUID.jpg files and return DataFrame with UUID and path columns."""
    logger.info(f"Scanning for <uuid>.jpg files in {input_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_data = []
    seen_uuids = set()

    # Find all .jpg files recursively
    for jpg_path in input_dir.rglob("*.jpg"):
        # Extract UUID from filename (remove .jpg extension)
        uuid_str = jpg_path.stem
        if uuid_str in seen_uuids:
            logger.warning(f"Duplicate UUID found in {input_dir}: {uuid_str=}")
            continue
        seen_uuids.add(uuid_str)
        image_data.append({C.UUID_STRING: uuid_str, "image_path": jpg_path})

    df = pd.DataFrame(image_data)
    df = df.set_index(C.UUID_STRING, verify_integrity=True)
    logger.info(f"Found {len(df)} .jpg files")
    return df


def _get_existing_embeddings(engine: Engine) -> set[str]:
    """Get set of UUIDs that already have embeddings in the database."""
    try:
        query = f"SELECT {C.UUID_STRING} FROM {EmbeddingsTables.IMAGE_EMBEDDINGS}"
        df = pd.read_sql(query, engine)
        return set(df[C.UUID_STRING].values)
    except Exception:
        # Table doesn't exist yet, return empty set
        return set()


def _write_embeddings_worker(
    write_queue: Queue[pd.DataFrame | None],
    engine: Engine,
) -> None:
    """Worker thread that writes embeddings to database using upsert_dataframe_to_sqlite."""

    try:
        while True:
            item = write_queue.get()
            if item is None:  # Sentinel value to stop
                break

            # Write the DataFrame directly to the database
            item[C.SEMANTIC_EMBEDDING] = item[C.SEMANTIC_EMBEDDING].map(
                serialize_numpy_array
            )
            upsert_dataframe_to_sqlite(item, engine, EmbeddingsTables.IMAGE_EMBEDDINGS)
            logger.debug(f"Wrote batch of {len(item)} embeddings to database")
    except Exception as e:
        logger.exception(
            f"Error writing embeddings to database.  Not trying to continue."
        )
        sys.exit(1)


def _compute_embeddings_batch(
    batch_df: pd.DataFrame,
    write_queue: Queue[pd.DataFrame | None],
    embedder: EmbeddingInterface,
) -> None:
    """Compute embeddings for a batch of images and enqueue them for writing."""
    try:
        # Compute embeddings for the batch using the embedder instance
        # Add embeddings directly to the DataFrame
        embedder.compute_image_embeddings(
            batch_df,
            image_path_column="image_path",
            output_embedding_column=C.SEMANTIC_EMBEDDING,
        )

        # Create output DataFrame with just UUID and embedding columns
        output_df = batch_df[[C.SEMANTIC_EMBEDDING]].copy()

        # Enqueue the DataFrame for writing
        write_queue.put(output_df)

    except Exception as e:
        logger.error(f"Error computing embeddings for batch: {e}")
        raise


def _load_embeddings_from_db(engine: Engine) -> pd.DataFrame:
    """Load all embeddings from database into a DataFrame."""

    try:
        query = f"""
        SELECT {C.UUID_STRING}, {C.SEMANTIC_EMBEDDING}
        FROM {EmbeddingsTables.IMAGE_EMBEDDINGS}
        """
        df = pd.read_sql(query, engine)

        if df.empty:
            return pd.DataFrame(columns=[C.UUID_STRING, C.SEMANTIC_EMBEDDING])

        # Deserialize embeddings using map operation
        df[C.SEMANTIC_EMBEDDING] = df[C.SEMANTIC_EMBEDDING].map(deserialize_numpy_array)

        return df
    except Exception:
        # Table doesn't exist yet, return empty DataFrame
        return pd.DataFrame(columns=[C.UUID_STRING, C.SEMANTIC_EMBEDDING])


def _store_umap_model(engine: Engine, umap_model: Any) -> None:
    """Store UMAP model in database."""

    # Serialize model with pickle
    model_blob = pickle.dumps(umap_model)
    df = pd.DataFrame(
        {
            C.UUID_STRING: [EmbeddingsTables.UMAP_MODEL],
            "MODEL": [model_blob],
        }
    )
    df = df.set_index(C.UUID_STRING, verify_integrity=True)
    upsert_dataframe_to_sqlite(df, engine, EmbeddingsTables.UMAP_MODEL)
    logger.info("Stored UMAP model in database")


def _upsert_umap_projections(engine: Engine, df: pd.DataFrame) -> None:
    """Upsert UMAP embeddings into the haversine table."""
    assert df.index.name == C.UUID_STRING
    df = df.dropna(
        subset=[C.SEMANTIC_EMBEDDING_UMAP_LATITUDE, C.SEMANTIC_EMBEDDING_UMAP_LONGITUDE]
    )
    df = df[
        [
            C.SEMANTIC_EMBEDDING_UMAP_LATITUDE,
            C.SEMANTIC_EMBEDDING_UMAP_LONGITUDE,
        ]
    ]
    upsert_dataframe_to_sqlite(
        df,
        engine,
        EmbeddingsTables.IMAGE_UMAP_PROJECTIONS,
    )
    logger.info(f"Upserted {len(df)} UMAP embeddings")


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

    # Validate input directory
    if not input_dir.exists():
        raise typer.BadParameter(f"Input directory does not exist: {input_dir}")

    # Create output directory if needed
    output_embeddings_db.parent.mkdir(parents=True, exist_ok=True)

    # Create database engine
    engine = create_engine(f"sqlite:///{output_embeddings_db}")

    # # Create database schema
    # umap_model_table = _create_database_schema(engine, embedding_type)

    # Scan for image files
    all_images_df = _scan_image_files(input_dir)

    if all_images_df.empty:
        logger.warning("No valid UUID.jpg files found in input directory")
        return

    # Determine which embeddings need to be computed
    if recompute_existing_embeddings:
        missing_images_df = all_images_df.copy()
        logger.info(f"Recomputing all {len(missing_images_df)} embeddings")
    else:
        existing_uuids = _get_existing_embeddings(engine)
        missing_images_df = all_images_df[
            ~all_images_df[C.UUID_STRING].isin(existing_uuids)
        ].copy()
        logger.info(f"Found {len(existing_uuids)} existing embeddings")
        logger.info(f"Need to compute {len(missing_images_df)} new embeddings")

    if missing_images_df.empty:
        logger.info("No embeddings need to be computed")
    else:
        # Create embedder instance once (efficient model loading)
        logger.info(f"Initializing embedder for {embedding_type}")
        embedder = _get_embedder_instance(embedding_type)

        # Create batches as DataFrame slices
        batches: list[pd.DataFrame] = []
        for i in range(0, len(missing_images_df), batch_size):
            batch_df = missing_images_df.iloc[i : i + batch_size].copy()
            batches.append(batch_df)

        # Set up database write queue and worker
        write_queue: Queue[pd.DataFrame | None] = Queue()

        # Start database writer thread
        writer_thread = threading.Thread(
            target=_write_embeddings_worker,
            args=(write_queue, engine),
            name="write_embeddings_worker",
        )

        writer_thread.start()
        try:
            # Compute embeddings in batches with progress bar
            logger.info(f"Computing embeddings in {len(batches)} batches")

            for batch_df in tqdm(batches, desc="Computing embeddings"):
                _compute_embeddings_batch(batch_df, write_queue, embedder)
        finally:
            # Then signal to stop and wait for thread to finish processing remaining items
            write_queue.put(None)  # Sentinel value
            logger.info("Waiting for all embeddings to be written to database...")
            writer_thread.join()

        logger.info("All embeddings written to database.")

    # Run UMAP projection if requested
    if reproject_umap_embeddings:
        logger.info("Loading embeddings for UMAP projection...")
        df = _load_embeddings_from_db(engine)
        df = df.set_index(C.UUID_STRING, verify_integrity=True)

        if len(df) < 5:
            logger.warning(
                f"{len(df)=} is too few embeddings found for UMAP projection."
            )
        else:
            logger.info(f"Running UMAP projection on {len(df)} embeddings...")

            # Run UMAP projection
            umap_model = umap_haversine.run_umap_projection(
                df,
                embedding_column=C.SEMANTIC_EMBEDDING,
                output_lat_column=C.SEMANTIC_EMBEDDING_UMAP_LATITUDE,
                output_lon_column=C.SEMANTIC_EMBEDDING_UMAP_LONGITUDE,
            )

            # Upsert the UMAP embeddings for the images into the haversine table
            _upsert_umap_projections(engine, df)

            # Store UMAP model in database
            _store_umap_model(engine, umap_model)
            logger.info("UMAP model stored in database")

    logger.info("Embedding computation completed successfully")


if __name__ == "__main__":
    app()
