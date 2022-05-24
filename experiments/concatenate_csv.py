import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Combine a bunch of CSV files into a single file",
    allow_abbrev=False,
)

parser.add_argument(
    '-i',
    '--input',
    nargs='+',
    action='store',
    required=True
)

parser.add_argument(
    '-o',
    '--output',
    action='store',
    required=True
)

args = parser.parse_args()

dataframes = [pd.read_csv(csv) for csv in args.input]
out_df = pd.concat(dataframes)
out_df.to_csv(args.output, index=False)

