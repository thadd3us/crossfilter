"""Temporal CDF Plot Module.

Simple temporal CDF plotting using Plotly Express.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from crossfilter.core.schema import SchemaColumns


def create_temporal_cdf(
    df: pd.DataFrame,
    title: str = "Temporal Distribution (CDF)",
    groupby: Optional[str] = None,
) -> go.Figure:
    """Create a Plotly CDF plot for temporal data."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            # xref="paper",
            # yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df = df.copy()
    time_column = SchemaColumns.TIMESTAMP_UTC

    df[SchemaColumns.DF_ID] = df.index

    # TODO: This could be a cacheable amount of work.
    if not groupby:
        groupby = "Data"
        df[groupby] = "All"
    df[groupby] = df[groupby].astype(str).fillna("Unknown")
    df["groupby_count"] = df.groupby(groupby)[SchemaColumns.DF_ID].transform("count")
    df["group_and_count"] = df[groupby] + " (" + df["groupby_count"].astype(str) + ")"
    df = df.sort_values(by=["groupby_count", groupby], ascending=[False, True])

    groups = []
    for _, group in df.groupby("group_and_count"):
        group = group.sort_values(by=time_column)
        group["CDF"] = np.arange(1, len(group) + 1) / len(group)
        groups.append(group)
    plot_df = pd.concat(groups)

    fig = px.scatter(
        plot_df,
        x=time_column,
        y="CDF",
        color="group_and_count",
        custom_data=[SchemaColumns.DF_ID],
        hover_data=[SchemaColumns.DF_ID, SchemaColumns.NAME, SchemaColumns.UUID_STRING],
        # hover_data={"label": True},
    )
    # fig.update_traces(
    #     hovertemplate="x: %{x}<br>ECDF: %{y}<br>Label: %{customdata[0]}<extra></extra>"
    # )

    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        hovermode="closest",
        showlegend=True,
        # height=400,
        # margin={"l": 50, "r": 50, "t": 50, "b": 50},
    )

    fig.update_xaxes(type="date", tickangle=45)  # tickformat="%Y-%m-%d %H:%M",

    return fig

    # elif SchemaColumns.TIMESTAMP_UTC not in df.columns:
    #     fig = go.Figure()
    #     fig.add_annotation(
    #         text="No timestamp data found",
    #         xref="paper",
    #         yref="paper",
    #         x=0.5,
    #         y=0.5,
    #         showarrow=False,
    #     )
    # else:
    #     # Create ECDF plot using plotly express
    #     temp_fig = px.ecdf(df, x=SchemaColumns.TIMESTAMP_UTC, title=title)

    #     # Extract the data and convert numpy arrays to lists before creating final figure
    #     if temp_fig.data:
    #         original_trace = temp_fig.data[0]

    #         # Convert arrays to lists to prevent binary encoding
    #         x_data = (
    #             original_trace.x.tolist()
    #             if hasattr(original_trace.x, "tolist")
    #             else list(original_trace.x)
    #         )
    #         y_data = (
    #             original_trace.y.tolist()
    #             if hasattr(original_trace.y, "tolist")
    #             else list(original_trace.y)
    #         )

    #         # Add customdata for interactivity using df_id (DataFrame index)
    #         customdata = [{"df_id": int(idx)} for idx in df.index]

    #         # Create new trace with list data
    #         new_trace = go.Scatter(
    #             x=x_data,
    #             y=y_data,
    #             mode="lines",
    #             line=dict(shape="hv"),
    #             customdata=customdata,
    #             hovertemplate=(
    #                 "<b>Time:</b> %{x}<br>"
    #                 "<b>Cumulative Probability:</b> %{y}<br>"
    #                 "<b>Row ID:</b> %{customdata.df_id}<br>"
    #                 "<extra></extra>"
    #             ),
    #         )

    #         # Create new figure with the converted trace
    #         fig = go.Figure(data=[new_trace])
    #     else:
    #         fig = go.Figure()

    # # Configure layout
    # fig.update_layout(
    #     title=title,
    #     xaxis_title="Time",
    #     yaxis_title="Cumulative Probability",
    #     hovermode="closest",
    #     showlegend=False,
    #     height=400,
    #     margin={"l": 50, "r": 50, "t": 50, "b": 50},
    # )

    # # Configure x-axis for time formatting
    # fig.update_xaxes(type="date", tickformat="%Y-%m-%d %H:%M", tickangle=45)

    # return fig.to_dict()
