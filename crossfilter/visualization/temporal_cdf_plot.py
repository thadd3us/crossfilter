"""Temporal CDF Plot Module.

This module provides functionality for creating temporal CDF (Cumulative Distribution Function) plots
using Plotly. The implementation follows a simple design:

- All DataFrames have integer indices (df_id) for frontend communication
- Whether aggregated or individual data, the DataFrame will always have an integer index
- For aggregated data, there will be a UUID_STRING column with an example instance from each bucket
- The frontend communicates selected elements using these integer indices (df_id)

No special cases are needed - the implementation handles both aggregated and individual data uniformly
by using the DataFrame's integer index as the primary identifier for selection communication.
"""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_temporal_cdf(df: pd.DataFrame, title: str = "Temporal Distribution (CDF)") -> dict[str, Any]:
    """Create a Plotly CDF plot for temporal data.

    Args:
        df: DataFrame with temporal data. Must have an integer index (df_id).
            For aggregated data, will have 'count' column and UUID_STRING column.
            For individual data, each row represents one data point.
        title: Plot title

    Returns:
        Plotly figure as dictionary
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    else:
        # Find timestamp column
        timestamp_col = None
        for col in df.columns:
            if col.startswith('QUANTIZED_TIMESTAMP') or col == 'TIMESTAMP_UTC':
                timestamp_col = col
                break

        if timestamp_col is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No timestamp data found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        else:
            # Handle aggregated data by expanding to individual points for px.ecdf
            if 'count' in df.columns:
                # Aggregated data - expand back to individual points
                expanded_timestamps = []
                expanded_df_ids = []
                
                for idx, row in df.iterrows():
                    timestamp = row[timestamp_col]
                    count = int(row['count'])
                    
                    # Add individual points for this timestamp, all referencing the same df_id
                    for _ in range(count):
                        expanded_timestamps.append(timestamp)
                        expanded_df_ids.append(int(idx))
                
                # Create DataFrame for px.ecdf
                plot_df = pd.DataFrame({
                    timestamp_col: expanded_timestamps,
                    'df_id': expanded_df_ids
                })
            else:
                # Individual data - use directly
                plot_df = df.copy()
                plot_df['df_id'] = df.index.astype(int)
            
            # Create ECDF plot
            fig = px.ecdf(plot_df, x=timestamp_col, title=title)
            
            # Add customdata for interactivity using df_id
            customdata = [{"df_id": df_id} for df_id in plot_df['df_id']]
            
            # Update trace with customdata and hover template
            if fig.data:
                trace = fig.data[0]
                # Convert numpy arrays to lists for JSON serialization
                trace.x = trace.x.tolist() if hasattr(trace.x, 'tolist') else trace.x
                trace.y = trace.y.tolist() if hasattr(trace.y, 'tolist') else trace.y
                trace.customdata = customdata
                
                # Custom hover template showing df_id
                trace.hovertemplate = (
                    '<b>Time:</b> %{x}<br>'
                    '<b>Cumulative Probability:</b> %{y}<br>'
                    '<b>Row ID:</b> %{customdata.df_id}<br>'
                    '<extra></extra>'
                )

    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Cumulative Probability",
        hovermode='closest',
        showlegend=False,
        height=400,
        margin={"l": 50, "r": 50, "t": 50, "b": 50}
    )

    # Configure x-axis for time formatting
    fig.update_xaxes(
        type='date',
        tickformat='%Y-%m-%d %H:%M',
        tickangle=45
    )

    return fig.to_dict()