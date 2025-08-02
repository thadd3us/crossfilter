from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSubset:
    subset_uuid: str

    df: pd.DataFrame
