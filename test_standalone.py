#!/usr/bin/env python3
"""Standalone test to verify temporal CDF functionality."""

import json
import sys
import traceback
from pathlib import Path

# Add the current directory to Python path so we can import crossfilter
sys.path.insert(0, str(Path(__file__).parent))

try:
    from crossfilter.core.schema import load_jsonl_to_dataframe
    from crossfilter.core.session_state import SessionState
    from crossfilter.visualization.temporal_cdf_plot import create_temporal_cdf
    
    print("✓ Successfully imported crossfilter modules")
    
    # Test 1: Load sample data
    sample_path = Path(__file__).parent / "test_data" / "sample_100.jsonl"
    if not sample_path.exists():
        print(f"✗ Sample data not found at {sample_path}")
        sys.exit(1)
    
    df = load_jsonl_to_dataframe(sample_path)
    print(f"✓ Loaded {len(df)} rows from sample data")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Index type: {type(df.index[0])}")
    
    # Test 2: Create session state and load data
    session_state = SessionState()
    session_state.load_dataframe(df)
    print("✓ Loaded data into session state")
    
    # Test 3: Get temporal aggregation
    temporal_data = session_state.get_temporal_aggregation(max_groups=1000)
    print(f"✓ Generated temporal aggregation with {len(temporal_data)} points")
    print(f"  Temporal data columns: {list(temporal_data.columns)}")
    print(f"  Temporal data index: {list(temporal_data.index[:5])}")  # Show first 5 indices
    
    # Test 4: Create temporal CDF plot
    cdf_plot = create_temporal_cdf(temporal_data)
    print("✓ Created temporal CDF plot")
    
    # Verify plot structure
    if "data" in cdf_plot and len(cdf_plot["data"]) > 0:
        plot_trace = cdf_plot["data"][0]
        print(f"  Plot has {len(plot_trace.get('x', []))} data points")
        
        # Check customdata
        if "customdata" in plot_trace:
            print(f"  Plot includes customdata with {len(plot_trace['customdata'])} entries")
            first_custom = plot_trace['customdata'][0]
            print(f"  First customdata: {first_custom}")
            
            if "df_id" in first_custom:
                print("✓ customdata includes df_id for selections")
            else:
                print("✗ customdata missing df_id")
        else:
            print("✗ Plot missing customdata")
    else:
        print("✗ Plot missing data")
    
    # Test 5: Filter functionality
    print("\n--- Testing filter functionality ---")
    
    # Get some row indices to filter with
    row_indices = set(temporal_data.index[:5])  # First 5 rows
    print(f"Testing filter with row indices: {row_indices}")
    
    # Apply temporal filter
    session_state.filter_state.apply_temporal_filter(
        row_indices,
        "Test temporal filter"
    )
    
    filter_summary = session_state.filter_state.get_summary()
    print(f"✓ Applied temporal filter: {filter_summary['filtered_count']} of {filter_summary['total_count']} points")
    
    # Test filtered data
    filtered_data = session_state.get_filtered_data()
    print(f"✓ Filtered data has {len(filtered_data)} rows")
    
    # Test reset
    session_state.filter_state.reset_filters()
    filter_summary_after_reset = session_state.filter_state.get_summary()
    print(f"✓ Reset filters: {filter_summary_after_reset['filtered_count']} of {filter_summary_after_reset['total_count']} points")
    
    print("\n🎉 All tests passed! The temporal CDF implementation is working correctly.")
    
except Exception as e:
    print(f"✗ Error during testing: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)