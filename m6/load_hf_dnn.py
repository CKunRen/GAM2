import argparse
import json

import numpy as np

from dnn_surrogate import load_checkpoint, predict_numpy


def main():
    parser = argparse.ArgumentParser(description="Load a trained DNN surrogate and run prediction.")
    parser.add_argument("--model", required=True, help="Path to best_dnn_surrogate.pt")
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file containing a 2D array with shape [n_samples, 52].",
    )
    args = parser.parse_args()

    bundle = load_checkpoint(args.model)
    with open(args.input_json, "r", encoding="utf-8") as f:
        x = np.asarray(json.load(f), dtype=np.float32)
    y = predict_numpy(bundle, x)
    print(json.dumps(y.tolist(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
