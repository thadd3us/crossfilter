from dataclasses import dataclass

import pandas as pd


@dataclass
class GroupedBucketedProjection:
    projection_uuid: str
    grouped_by_columns: list[str]

    projected_data: pd.DataFrame
