"""Utilities for working with Lightroom sqlite3 databases."""

import dataclasses
import enum
import glob
import logging
import os
import pathlib
import sqlite3
import urllib
import zipfile
from typing import Optional, Set

import dataclasses_json

import maya
import pandas as pd

import thad.lightroom.db_reader.pandas_util as pandas_util


@dataclasses.dataclass
class LightroomDb(object):
    images_df: Optional[pd.DataFrame] = None
    keywords_df: Optional[pd.DataFrame] = None
    collections_df: Optional[pd.DataFrame] = None


@dataclasses.dataclass
class LoadDbConfig(dataclasses_json.DataClassJsonMixin):
    ignore_collections: Set[str] = dataclasses.field(default_factory=set)


DEFAULT_LOAD_DB_CONFIG_JSON = """
{
  "ignore_collections": [
    "quick collection"
  ]
}
"""
DEFAULT_LOAD_DB_CONFIG = LoadDbConfig.from_json(DEFAULT_LOAD_DB_CONFIG_JSON)


def load_db(path: str, config=DEFAULT_LOAD_DB_CONFIG) -> LightroomDb:
    logging.info("load_db, path=%s", path)
    if not os.path.exists(path):
        raise ValueError(f"No file at {path}")

    lightroom_db = LightroomDb()

    connection = sqlite3.connect(path)
    try:
        cursor = connection.cursor()
        lightroom_db.images_df = query_to_data_frame(cursor, QUERY_IMAGES)
        lightroom_db.images_df[Column.PARSED_CAPTURE_TIME.value] = (
            lightroom_db.images_df[Column.CAPTURE_TIME.value].map(parse_date_time)
        )

        lightroom_db.images_df[Column.GPS_LOCATION.value] = lightroom_db.images_df[
            [Column.GPS_LATITUDE.value, Column.GPS_LONGITUDE.value]
        ].aggregate(
            lambda x: pandas_util.apply_if_none_null(x, str, None),
            # args=(str, None),
            axis="columns",
            raw=True,
            # result_type="reduce",
        )

        def image_link(x):
            filename = os.path.join(*x.tolist())
            escaped = urllib.parse.quote(filename, safe="/")
            return f"file://{escaped}"

        lightroom_db.images_df[Column.IMAGE_LINK.value] = lightroom_db.images_df[
            [Column.ROOT_PATH.value, Column.PATH_FROM_ROOT.value, Column.FILENAME.value]
        ].apply(image_link, axis="columns", raw=True)

        # Not using the index, just checking integrity.
        lightroom_db.images_df.set_index(Column.ID_GLOBAL.value, verify_integrity=True)

        keywords_df = query_to_data_frame(cursor, QUERY_KEYWORDS)
        keywords_df = keywords_df.merge(
            lightroom_db.images_df,
            how="left",
            left_on=Column.ID_GLOBAL.value,
            right_on=Column.ID_GLOBAL.value,
            suffixes=("", ""),
        )
        lightroom_db.keywords_df = keywords_df
        keyword_series = keywords_df.groupby(Column.ID_GLOBAL.value)[
            Column.KEYWORD.value
        ].apply(lambda x: tuple(sorted(x)))
        lightroom_db.images_df = lightroom_db.images_df.merge(
            keyword_series, how="left", left_on=Column.ID_GLOBAL.value, right_index=True
        )

        collections_df = query_to_data_frame(cursor, QUERY_COLLECTIONS)
        collections_df = collections_df[
            ~collections_df[Column.COLLECTION.value].isin(config.ignore_collections)
        ]
        collections_df = collections_df.merge(
            lightroom_db.images_df,
            how="left",
            left_on=Column.ID_GLOBAL.value,
            right_on=Column.ID_GLOBAL.value,
            suffixes=("", ""),
        )
        lightroom_db.collections_df = collections_df
        collection_series = collections_df.groupby(Column.ID_GLOBAL.value)[
            Column.COLLECTION.value
        ].apply(lambda x: tuple(sorted(x)))
        lightroom_db.images_df = lightroom_db.images_df.merge(
            collection_series,
            how="left",
            left_on=Column.ID_GLOBAL.value,
            right_index=True,
        )

    finally:
        connection.close()
    return lightroom_db


@enum.unique
class Column(enum.StrEnum):
    ID_GLOBAL = "Adobe_images_id_global"
    ROOT_FILE = "Adobe_images_rootFile"  # Pointer to the file index.

    FILENAME = "AgLibraryFile_idx_filename"
    PATH_FROM_ROOT = "AgLibraryFolder_pathFromRoot"
    ROOT_PATH = "AgLibraryRootFolder_absolutePath"
    IMAGE_LINK = "IMAGE_LINK"

    GPS_LATITUDE = "AgHarvestedExifMetadata_gpsLatitude"
    GPS_LONGITUDE = "AgHarvestedExifMetadata_gpsLongitude"
    RATING = "Adobe_images_rating"
    COLOR_LABELS = "Adobe_images_colorLabels"
    CAPTURE_TIME = "Adobe_images_captureTime"
    HASH = "AgLibraryFile_importHash"
    CAPTION = "AgLibraryIPTC_caption"

    GPS_LOCATION = "GPS_LOCATION"
    PARSED_CAPTURE_TIME = "PARSED_CAPTURE_TIME"

    # Not present for images; just in specific DataFrames.
    KEYWORD = "AgLibraryKeyword_name"
    COLLECTION = "AgLibraryCollection_name"


TABLE_MARKER_PREFIX = "TABLE_MARKER_"

QUERY_SNIPPET_SELECT_IMAGE_LOCATION = f"""
    0 AS {TABLE_MARKER_PREFIX}Adobe_images,
    Adobe_images.*,
    0 AS {TABLE_MARKER_PREFIX}AgLibraryFile,
    AgLibraryFile.*,
    0 AS {TABLE_MARKER_PREFIX}AgLibraryFolder,
    AgLibraryFolder.*,
    0 AS {TABLE_MARKER_PREFIX}AgLibraryRootFolder,
    AgLibraryRootFolder.*
"""

QUERY_SNIPPET_JOIN_IMAGE_LOCATION = """
LEFT JOIN AgLibraryFile ON AgLibraryFile.id_local = Adobe_images.rootFile
LEFT JOIN AgLibraryFolder ON AgLibraryFolder.id_local = AgLibraryFile.folder
LEFT JOIN AgLibraryRootFolder ON AgLibraryRootFolder.id_local = AgLibraryFolder.rootFolder
"""

# Using *s below since I don't know the DB schema well, and want to notice
# if new things appear.
# Using dummy columns to mark which columns come from which tables.
QUERY_IMAGES = f"""
SELECT
    0 AS {TABLE_MARKER_PREFIX}AgLibraryIPTC,
    AgLibraryIPTC.*,
    0 AS {TABLE_MARKER_PREFIX}AgHarvestedExifMetadata,
    AgHarvestedExifMetadata.*,
    {QUERY_SNIPPET_SELECT_IMAGE_LOCATION}
FROM Adobe_images
LEFT JOIN AgLibraryIPTC ON AgLibraryIPTC.image = Adobe_images.id_local
LEFT JOIN AgHarvestedExifMetadata ON AgHarvestedExifMetadata.image = Adobe_images.id_local
{QUERY_SNIPPET_JOIN_IMAGE_LOCATION}
;
"""

QUERY_KEYWORDS = f"""
SELECT
    0 AS {TABLE_MARKER_PREFIX}AgLibraryKeywordImage,
    AgLibraryKeywordImage.*,
    0 AS {TABLE_MARKER_PREFIX}AgLibraryKeyword,
    AgLibraryKeyword.*,
    0 AS {TABLE_MARKER_PREFIX}Adobe_images,
    Adobe_images.id_global
FROM AgLibraryKeywordImage
LEFT JOIN Adobe_images ON Adobe_images.id_local = AgLibraryKeywordImage.image
LEFT JOIN AgLibraryKeyword ON AgLibraryKeyword.id_local = AgLibraryKeywordImage.tag
;
"""

QUERY_COLLECTIONS = f"""
SELECT
    0 AS {TABLE_MARKER_PREFIX}AgLibraryCollectionImage,
    AgLibraryCollectionImage.*,
    0 AS {TABLE_MARKER_PREFIX}AgLibraryCollection,
    AgLibraryCollection.*,
    0 AS {TABLE_MARKER_PREFIX}Adobe_images,
    Adobe_images.id_global
FROM AgLibraryCollectionImage
LEFT JOIN Adobe_images ON Adobe_images.id_local = AgLibraryCollectionImage.image
LEFT JOIN AgLibraryCollection ON AgLibraryCollection.id_local = AgLibraryCollectionImage.collection
;
"""


def query_to_data_frame(cursor: sqlite3.Cursor, query: str) -> pd.DataFrame:
    cursor.execute(query)
    rows = cursor.fetchall()
    column_names = []
    table_marker_columns = []
    table_prefix = ""
    for d in cursor.description:
        name = d[0]
        if name.startswith(TABLE_MARKER_PREFIX):
            # This is a special marker column.
            table_prefix = name[len(TABLE_MARKER_PREFIX) :] + "_"
            column_names.append(name)
            table_marker_columns.append(name)
        else:
            column_names.append(table_prefix + name)
    df = pd.DataFrame.from_records(rows, columns=column_names)
    df = df.drop(labels=table_marker_columns, axis=1)
    return df


def parse_date_time(date_time_str: Optional[str]) -> Optional[maya.MayaDT]:
    if date_time_str is None:
        return None
    try:
        return maya.parse(date_time_str)
    except ValueError as e:
        logging.error("Unable to parse time: %s\n%s", date_time_str, e)
    return None


def maybe_unzip(filename: str) -> str:
    """If a catalog is a zip file, extract to a known cache location."""
    if isinstance(filename, pathlib.Path):
        filename = str(filename)
    if not filename.endswith(".zip"):
        logging.info("Not a zipfile, no need to extract.")
        return filename
    filename_without_slash = filename.replace("/", "_")
    dest_dir = f"/tmp/{filename_without_slash}"
    if not os.path.exists(dest_dir):
        logging.info("Extracting to %s", dest_dir)
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(dest_dir)
    else:
        logging.info("Already extracted.")

    dest_dir_contents = glob.glob(os.path.join(dest_dir, "*.lrcat"))
    assert len(dest_dir_contents) == 1, dest_dir_contents
    return os.path.join(dest_dir, dest_dir_contents[0])
