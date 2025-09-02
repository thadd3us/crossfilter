import sqlite3
import pandas as pd
import plotly.express as px
import typer
from pathlib import Path
from typing import Optional


app = typer.Typer(help="SQLite Space Estimator (Tables, Columns, Indices)")


def get_table_list(conn):
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]


def get_index_list(conn):
    return [
        (row[1], row[0])
        for row in conn.execute(
            "SELECT tbl_name, name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
    ]


def estimate_column_sizes(conn, table, row_count, sample_size=1000):
    """Estimate total column sizes by sampling data and scaling."""
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {sample_size}", conn)
    except Exception:
        return []

    if df.empty or row_count == 0:
        return []

    column_sizes = []
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            avg_size = 0
        elif pd.api.types.is_numeric_dtype(sample):
            avg_size = sample.memory_usage(index=False) / len(sample)
        else:
            avg_size = sample.astype(str).str.encode("utf-8").map(len).mean()
        est_total = avg_size * row_count
        column_sizes.append((table, col, est_total))
    return column_sizes


@app.command()
def analyze(
    sqlite_file: Path = typer.Argument(
        ..., exists=True, help="Path to SQLite database"
    ),
    output_html: Optional[Path] = typer.Option(
        "treemap.html", help="Output HTML file for treemap"
    ),
):
    conn = sqlite3.connect(str(sqlite_file))
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    total_db_size = page_size * page_count

    typer.echo(f"🔍 Database size: {total_db_size:,} bytes")

    tables = get_table_list(conn)
    indices = get_index_list(conn)

    rows = []
    column_rows = []

    for table in tables:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        # Estimate full table size by sampling
        try:
            sample_df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1000", conn)
            if sample_df.empty or row_count == 0:
                avg_row_size = 0
            else:
                total_bytes = sample_df.memory_usage(deep=True).sum()
                avg_row_size = total_bytes / len(sample_df)
            est_table_size = avg_row_size * row_count
        except Exception:
            est_table_size = 0
            avg_row_size = 0
            row_count = 0

        rows.append(
            dict(type="table", table=table, name=table, parent="", bytes=est_table_size)
        )

        # Estimate per-column sizes
        column_estimates = estimate_column_sizes(conn, table, row_count)
        for tbl, col, col_bytes in column_estimates:
            column_rows.append(
                dict(type="column", table=tbl, name=col, parent=tbl, bytes=col_bytes)
            )

    # Add index estimates (very rough, just assume 10% of table size or minimum 1KB)
    for tbl_name, idx_name in indices:
        # Find estimated table size to scale index estimate
        tbl_size = next((r["bytes"] for r in rows if r["name"] == tbl_name), 0)
        index_bytes = max(1024, tbl_size * 0.1)  # crude estimate
        rows.append(
            dict(
                type="index",
                table=tbl_name,
                name=idx_name,
                parent="",
                bytes=index_bytes,
            )
        )

    all_rows = rows + column_rows
    df = pd.DataFrame(all_rows)

    typer.echo("\n--- Estimated Space Breakdown ---")
    typer.echo(
        df[["type", "table", "name", "bytes"]]
        .sort_values("bytes", ascending=False)
        .to_string(index=False)
    )

    fig = px.treemap(
        df,
        path=["type", "parent", "name"],
        values="bytes",
        title=f"SQLite Space Usage (Estimated): {sqlite_file.name}",
    )
    fig.write_html(str(output_html))
    typer.echo(f"\n✅ Treemap written to: {output_html.resolve()}")


if __name__ == "__main__":
    app()
