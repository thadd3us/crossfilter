from dataclasses import dataclass
from typing import Any, Final
import uuid
from crossfilter.core.data_subset import DataSubset

import pandas as pd


@dataclass
class _Group:
    group_key: tuple[Any, ...]
    group_df: pd.DataFrame
    unbucketed_row_count: int = -1
    optional_bucketed_on_column: str | None = None

    def __post_init__(self) -> None:
        self.unbucketed_row_count = len(self.group_df)


def _maybe_bucket_group(
    group: _Group,
    max_entries: int,
    bucketing_candidate_columns: tuple[str, ...],
) -> None:
    if len(group.group_df) <= max_entries:
        return

    raise NotImplementedError("Not implemented")
    # for bucketing_candidate_column in bucketing_candidate_columns:
    #     if group.group_df[bucketing_candidate_column].nunique() <= max_entries:
    #         group.bucketed_on_column = bucketing_candidate_column
    #         bucketed_df = group.group_df.groupby(
    #             bucketing_candidate_column, dropna=False
    #         )


@dataclass
class GroupedBucketedProjection:
    projection_uuid: str

    # Each group may be bucketed by a different column, depending on how many rows are in the group.
    grouped_by_columns: tuple[str, ...]

    # Key: how is this indexed?  Can it be (group_key)->bucket_key|uuid?
    projected_data: pd.DataFrame


def create_grouped_bucketed_projection(
    ds: DataSubset,
    group_by_columns: tuple[str, ...],
    # Determines whether we need to bucket.
    max_entries_per_group: int,
    bucketing_candidate_columns: tuple[str, ...],
    sort_groups_by_cardinality: bool,
) -> GroupedBucketedProjection:

    groups: list[_Group] = []
    if group_by_columns:
        for group_key, group_df in ds.df.groupby(group_by_columns, dropna=False):
            group = _Group(group_key=group_key, group_df=group_df)
            groups.append(group)
    else:
        group = _Group(group_key=(), group_df=ds.df)
        groups.append(group)

    for group in groups:
        _maybe_bucket_group(group, max_entries_per_group, bucketing_candidate_columns)

    if sort_groups_by_cardinality:
        groups.sort(key=lambda x: (-x.unbucketed_row_count, x.group_key))
    else:
        groups.sort(key=lambda x: x.group_key)

    return GroupedBucketedProjection(
        projection_uuid=str(uuid.uuid4()),
        grouped_by_columns=group_by_columns,
        projected_data=pd.concat(
            [group.group_df for group in groups], ignore_index=False
        ),
    )
    # groups: list[_Group] = []

    # group_key_to_group_df: dict[tuple[Any, ...], pd.DataFrame] = {}
    # group_key_to_group_size: dict[tuple[Any, ...], int] = {}
    # for group_name, group_df in groups:
    #     group_key = tuple(group_name)
    #     group_key_to_group_df[group_key] = group_df
    #     group_key_to_group_size[group_key] = len(group_df)

    # if sort_groups_by_cardinality:
    #     group_keys = sorted(
    #         group_key_to_group_size.keys(), key=lambda x: group_key_to_group_size[x]
    #     )
