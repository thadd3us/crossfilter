# Lightroom Test Catalogs

This directory contains test Lightroom catalog files (`.lrcat`) that are SQLite databases used for testing timezone handling and other functionality.

## Timezone Test Data State

### test_parse_timezone_aware_timestamps.lrcat
The `test_parse_timezone_aware_timestamps/test_parse_timezone_aware_timestamps.lrcat` database has been created specifically for testing timezone parsing with NULL values. It includes various timezone scenarios in the `Adobe_images` table's `captureTime` column:

| id_local | captureTime | Timezone Type | Description |
|----------|-------------|---------------|-------------|
| 86 | `2007-01-01T16:41:36` | Naive | No timezone information (assumed UTC) |
| 87 | `2020-06-01T01:45:58Z` | UTC | Timezone-aware UTC (Z suffix) |
| 88 | `2020-06-01T01:49:28+01:00` | +01:00 | Timezone-aware +1 hour (Central European Time) |
| 89 | `2020-06-01T01:50:02-07:00` | -07:00 | Timezone-aware -7 hours (Pacific Daylight Time) |
| 90 | `2020-06-01T01:51:36` | Naive | No timezone information (assumed UTC) |
| 91 | `NULL` | NULL | NULL captureTime value |

### test_catalog_00/test_catalog_fresh.lrcat
The original test catalog also has timezone test data (same as rows 86-90 above) but without the NULL case.

## CLI Command to View Timezone Data

To inspect the current timezone test data in the dedicated timezone test catalog:

```bash
python3 -c "
import sqlite3

conn = sqlite3.connect('test_data/lightroom/test_catalogs/test_parse_timezone_aware_timestamps/test_parse_timezone_aware_timestamps.lrcat')
cursor = conn.cursor()
cursor.execute('SELECT id_local, captureTime FROM Adobe_images ORDER BY id_local')
rows = cursor.fetchall()
conn.close()

print('Adobe_images captureTime values:')
for row in rows:
    print('  {}: {}'.format(row[0], row[1]))
"
```

To inspect the original test catalog (without NULL case):

```bash
python3 -c "
import sqlite3

conn = sqlite3.connect('test_data/lightroom/test_catalogs/test_catalog_00/test_catalog_fresh.lrcat')
cursor = conn.cursor()
cursor.execute('SELECT id_local, captureTime FROM Adobe_images ORDER BY id_local')
rows = cursor.fetchall()
conn.close()

print('Adobe_images captureTime values:')
for row in rows:
    print('  {}: {}'.format(row[0], row[1]))
"
```

## Purpose

This test data is designed to verify that:

1. **Naive timestamps** (without timezone info) are correctly handled and assumed to be UTC
2. **UTC timestamps** (with 'Z' suffix) are properly parsed as UTC
3. **Positive timezone offsets** (like +01:00) are correctly converted to UTC
4. **Negative timezone offsets** (like -07:00) are correctly converted to UTC
5. **NULL captureTime values** are properly handled without causing parsing errors
6. The parser correctly populates both `TIMESTAMP_MAYBE_TIMEZONE_AWARE` and `TIMESTAMP_UTC` columns