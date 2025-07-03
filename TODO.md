1. Address this problem that I see when starting the server:
```
/user_home/workspace/crossfilter/core/bucketing.py:161: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
  timestamps.dt.normalize().dt.to_period("M").dt.start_time
/user_home/workspace/crossfilter/core/bucketing.py:166: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
  timestamps.dt.normalize().dt.to_period("Y").dt.start_time
```
1. 