#!/usr/bin/env python3

"""Script that generates artificial random labeled data.

The generated data is meant to be used for testing machine learning algorithms.

The data consists of N labeled examples. Each example has D features and a binary label. The
features are generated randomly from a uniform distribution in [-1, 1]. The labels are generated
according to the logistic regression model. For details of the logistic regression model,
please see: https://en.wikipedia.org/wiki/Logistic_regression. The coefficients of the model are
generated randomly from a uniform distribution in [-1, 1].

The generated data is written to a CSV file. The examples correspond to rows in the CSV file. The
first D columns correspond to the features and the last column corresponds to the label.

Sample usage:

    python generate_artificial_data.py \
        --number-of-features 10 \
        --number-of-examples 100000 \
        --seed 12345 \
        --output-file data.csv
"""

from __future__ import annotations

import argparse

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-of-features", type=int, required=True, help="Number of features")
    parser.add_argument("--number-of-examples", type=int, required=True, help="Number of examples")
    parser.add_argument("--output-file", type=str, required=True, help="Name of the output file")
    parser.add_argument("--seed", type=int, required=False, default=None, help="Random seed")
    return parser.parse_args()


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Computes the sigmoid function.

    For more details, please see: https://en.wikipedia.org/wiki/Sigmoid_function.
    """
    return 1.0 / (1.0 + np.exp(-x))


def generate_data(
    number_of_examples: int,
    number_of_features: int,
    seed: int | None,
) -> np.ndarray:
    """Generates artificial labeled data."""
    random_number_generator = np.random.default_rng(seed)

    # Generate random input features.
    features = random_number_generator.uniform(
        low=-1.0,
        high=1.0,
        size=(number_of_examples, number_of_features),
    )

    # Generate coefficients for the model.
    model = random_number_generator.uniform(low=-1.0, high=1.0, size=number_of_features)

    # Generate the labels according to the model.
    log_odds = np.dot(features, model)
    probabilities = sigmoid(log_odds)
    labels = random_number_generator.binomial(n=1, p=probabilities)

    # Concatenate features and labels into a single array.
    return np.concatenate((features, labels.reshape(-1, 1)), axis=1)


def run() -> None:
    """Entry point of the script."""
    args = parse_args()
    data = generate_data(args.number_of_examples, args.number_of_features, args.seed)
    np.savetxt(args.output_file, data, delimiter=",", fmt="%12.8f")


if __name__ == "__main__":
    run()
