from io import StringIO
import pytest
import pandas as pd
from syrupy.assertion import SnapshotAssertion

from crossfilter.core.data_subset import DataSubset
from crossfilter.core.projection import (
    create_grouped_bucketed_projection,
    groupby_fillna,
    _C,
)
from crossfilter.core.schema import DataSchema as C


@pytest.fixture
def ds() -> DataSubset:
    csv_data = f"""{C.UUID_STRING},{C.DATA_TYPE},{C.RATING_0_TO_5},{C.NAME},{C.TIMESTAMP_UTC},TIMESTAMP_YEAR,TIMESTAMP_DAY,H3_01,H3_02
uuid0,GPS,1,name0,2024-07-27T9:00:00,2024-01-01,2024-07-27,abcd,abcd1
uuid1,GPS,2,name1,2024-07-28T9:00:00,2024-01-01,2024-07-28,abcd,abcd1
uuid2,GPS,3,name2,2024-07-29T9:00:00,2024-01-01,2024-07-29,abcd,abcd2
uuid3,GPS,4,name3,2024-07-30T9:00:00,2024-01-01,2024-07-30,abcd,abcd2
uuid4,PHOTO,5,name4,2024-07-31T9:00:00,2024-01-01,2024-07-31,abce,abce1
uuid5,PHOTO,,name5,2025-08-01T9:00:00,2025-01-01,2025-08-01,abce,abce1
uuid6,,,name6,2025-08-02T9:00:00,2025-01-01,2025-08-02,abce,abce2
uuid7,PHOTO,2,name7,2025-08-03T9:00:00,2025-01-01,2025-08-03,abce,abce2
uuid8,VIDEO,3,name8,2025-08-04T9:00:00,2025-01-01,2025-08-04,abce,abce3
uuid9,,4,name9,,,,abcd,abcde 
uuid10,VIDEO,5,name10,2025-08-06T9:00:00,2025-01-01,2025-08-06,, 
"""
    df = pd.read_csv(StringIO(csv_data))
    df[C.RATING_0_TO_5] = df[C.RATING_0_TO_5].astype(pd.Int32Dtype())
    # df = df.set_index(C.UUID_STRING, verify_integrity=True)
    return DataSubset(subset_uuid="test_subset_uuid", df=df)


def test_create_grouped_bucketed_projection_no_grouping_no_bucketing(
    ds: DataSubset,
    snapshot: SnapshotAssertion,
) -> None:
    projection = create_grouped_bucketed_projection(
        ds,
        group_by_columns=[],
        bucketing_candidate_columns=(),
        max_entries_per_group=100,
        sort_groups_by_cardinality=False,
    )
    assert projection.projected_data.to_dict(index=True, orient="records") == snapshot


def test_create_grouped_bucketed_projection_group_by_data_type_no_bucketing(
    ds: DataSubset,
    snapshot: SnapshotAssertion,
) -> None:
    projection = create_grouped_bucketed_projection(
        ds,
        group_by_columns=[C.DATA_TYPE],
        bucketing_candidate_columns=(),
        max_entries_per_group=100,
        sort_groups_by_cardinality=False,
    )
    assert projection.projected_data.to_dict(index=True, orient="records") == snapshot
    assert projection.projected_data[_C.GROUP_NAME].value_counts(
        dropna=False
    ).to_dict() == {"GPS": 4, "PHOTO": 3, "?": 2, "VIDEO": 2}


def test_create_grouped_bucketed_projection_group_by_data_type_bucketing_H3_COARSE(
    ds: DataSubset,
    snapshot: SnapshotAssertion,
) -> None:
    ds.df = ds.df.dropna(subset=["H3_01", "H3_02"])
    projection = create_grouped_bucketed_projection(
        ds,
        group_by_columns=[C.DATA_TYPE],
        bucketing_candidate_columns=("H3_02", "H3_01"),
        max_entries_per_group=1,
        sort_groups_by_cardinality=False,
    )
    assert projection.projected_data.to_dict(index=True, orient="records") == snapshot
    name_to_bucket_level = {
        g.group_name: g.optional_bucketed_on_column
        for g in projection.name_to_group.values()
    }
    assert name_to_bucket_level == {
        "?": "H3_01",
        "GPS": "H3_01",
        "PHOTO": "H3_01",
        "VIDEO": None,
    }

    projection = create_grouped_bucketed_projection(
        ds,
        group_by_columns=[C.DATA_TYPE],
        bucketing_candidate_columns=("H3_02", "H3_01"),
        max_entries_per_group=2,
        sort_groups_by_cardinality=False,
    )
    assert projection.projected_data.to_dict(index=True, orient="records") == snapshot
    name_to_bucket_level = {
        g.group_name: g.optional_bucketed_on_column
        for g in projection.name_to_group.values()
    }
    assert name_to_bucket_level == {
        "?": None,
        "GPS": "H3_02",
        "PHOTO": "H3_02",
        "VIDEO": None,
    }


def test_groupby_fillna(snapshot: SnapshotAssertion) -> None:
    df = pd.DataFrame({"A": [None, 2, 3], "B": [4, 5, None], "C": ["7", 8, None]})
    df["A"] = df["A"].astype(pd.Int64Dtype())
    new_df = groupby_fillna(df)
    assert new_df.to_dict(orient="records") == snapshot
    assert new_df["A"].tolist() == ["?", "2", "3"], "Don't be floats!"
