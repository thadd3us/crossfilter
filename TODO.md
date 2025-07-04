# Address this problem that I see when starting the server and when running the unit tests:
```
/user_home/workspace/crossfilter/core/bucketing.py:161: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
  timestamps.dt.normalize().dt.to_period("M").dt.start_time
/user_home/workspace/crossfilter/core/bucketing.py:166: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
  timestamps.dt.normalize().dt.to_period("Y").dt.start_time
```

# The web UI status line should be formatted like: "Status: # rows loaded with # columns (# MB), # remain in current view"
* Report the deep memory used by the biggest pandas df.
* You'll need to change some of the unit tests to support this.

