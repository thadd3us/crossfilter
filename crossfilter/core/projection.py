from dataclasses import dataclass
import logging
from typing import Any, Final
import uuid
from crossfilter.core.data_subset import DataSubset
import enum
from crossfilter.core.schema import SchemaColumns as C

import pandas as pd

logger = logging.getLogger(__name__)


# Need to have the same type when comparing NA and present values.
GROUPBY_NA_FILL_VALUE = "?"


class _C(enum.StrEnum):
    GROUP_NAME = "_GROUP_NAME"


def groupby_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in the groupby column with a placeholder value."""
    return df.astype(object).fillna(GROUPBY_NA_FILL_VALUE).astype(str)


@dataclass
class _Group:
    group_name: str
    group_df: pd.DataFrame
    unbucketed_row_count: int = -1
    optional_bucketed_on_column: str | None = None

    def __post_init__(self) -> None:
        self.unbucketed_row_count = len(self.group_df)


@dataclass
class GroupedBucketedProjection:
    projection_uuid: str

    # Each group may be bucketed by a different column, depending on how many rows are in the group.
    grouped_by_columns: list[str, ...]

    # Key: how is this indexed?  Can it be (group_key)->bucket_key|uuid?
    projected_data: pd.DataFrame

    name_to_group: dict[str, _Group]


def create_grouped_bucketed_projection(
    ds: DataSubset,
    group_by_columns: list[str, ...],
    # Determines whether we need to bucket.
    max_entries_per_group: int,
    bucketing_candidate_columns: tuple[str, ...],
    sort_groups_by_cardinality: bool,
) -> GroupedBucketedProjection:

    groups: list[_Group] = []
    if group_by_columns:
        df = ds.df.copy()

        df.loc[:, group_by_columns] = groupby_fillna(df.loc[:, group_by_columns])
        df[_C.GROUP_NAME] = df[group_by_columns].apply(
            lambda x: ", ".join(x), axis="columns"
        )

        for group_name, group_df in df.groupby(_C.GROUP_NAME, dropna=False):
            group = _Group(group_name=group_name, group_df=group_df)
            groups.append(group)
    else:
        group = _Group(group_name="", group_df=ds.df)
        groups.append(group)

    for group in groups:
        _maybe_bucket_group(group, max_entries_per_group, bucketing_candidate_columns)

    if sort_groups_by_cardinality:
        groups.sort(key=lambda x: (-x.unbucketed_row_count, x.group_name))
    else:
        groups.sort(key=lambda x: x.group_name)

    result = GroupedBucketedProjection(
        projection_uuid=str(uuid.uuid4()),
        grouped_by_columns=group_by_columns,
        projected_data=pd.concat(
            [group.group_df for group in groups], ignore_index=False
        ),
        name_to_group={g.group_name: g for g in groups},
    )
    return result


def _maybe_bucket_group(
    group: _Group,
    max_entries: int,
    bucketing_candidate_columns: tuple[str, ...],
) -> None:
    df = group.group_df
    original_count = len(df)
    if len(df) <= max_entries:
        logger.info(f"{group.group_name=}: {len(df)=} <= {max_entries=}, not bucketing")
        return
    if not bucketing_candidate_columns:
        logger.warning(
            f"{group.group_name=}: Want to bucket {len(df)=}, but have no columns."
        )
        return

    last_count = len(group.group_df)
    logger.info(f"{group.group_name=}: Looking for bucketing scheme for {len(df)=}")
    for c in bucketing_candidate_columns:
        assert df[c].notna().all()
        this_count = df[c].nunique()
        # This assert could fire because H3 levels are not perfectly hierarchical.
        assert this_count <= last_count, f"{this_count=} > {last_count=} for {c=}"
        if this_count <= max_entries:
            break
    logger.info(
        f"{group.group_name=}: Using bucketing scheme on {c=} with {this_count=}"
    )
    group.optional_bucketed_on_column = c

    df[C.COUNT] = df.groupby(group.optional_bucketed_on_column).transform("size")
    # Keep the first.
    df = df.drop_duplicates(subset=[group.optional_bucketed_on_column])
    assert len(df) == this_count, f"{len(df)=} != {this_count=}, {c=}"
    df = df.reset_index()
    df.index.name = C.DF_ID
    sum_count = df[C.COUNT].sum()
    assert sum_count == original_count, f"{sum_count=} != {original_count=}, {c=}"

    group.group_df = df
