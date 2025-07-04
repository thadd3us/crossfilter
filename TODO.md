# TODO Items

All previous TODO items have been completed:

✅ **Fixed timezone warnings in bucketing.py** - Fixed `dt.to_period()` calls that dropped timezone information by temporarily converting to timezone-naive with `dt.tz_convert(None)`, applying the period conversion, then restoring UTC timezone with `dt.tz_localize("UTC")`. Added assertions that enforce timezone-aware UTC timestamps and regression test that fails if the warning returns.

✅ **Updated web UI status format** - Changed status line to format: "Status: # rows loaded with # columns (# MB), # remain in current view" with proper memory usage reporting.

✅ **Added deep memory usage reporting** - Added `memory_usage_mb` field to session state broadcasting and session API endpoint using `df.memory_usage(deep=True)`.

✅ **Updated unit tests** - Updated all test snapshots and frontend tests to support the new status format and timezone-aware temporal columns.

No outstanding TODO items remain.