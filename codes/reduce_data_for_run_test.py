"""
reduce data size for run test
load all parquet data and overwrite by head(1000)
"""

import polars as pl
import glob

for path in [
    "otto/inputs/train_valid/train_parquet/*", 
    "otto/inputs/train_valid/test_parquet/*",
    "otto_submit/inputs/train_valid/train_parquet/*",
    "otto_submit/inputs/train_valid/test_parquet/*"
    ]:
    for d in glob.glob(path):
        df = pl.read_parquet(d)
        _df = df.head(1000)
        print("{} : {} -> {}".format(d, df.shape, _df.shape))
        _df.write_parquet(d)
