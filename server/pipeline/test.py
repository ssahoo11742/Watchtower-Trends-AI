import argparse
parser = argparse.ArgumentParser(description="Run the pipeline with custom parameters.")
parser.add_argument("-d", "--depth", type=int, default=1, help="Depth of topic modeling.")
args = parser.parse_args()
print(args.depth)