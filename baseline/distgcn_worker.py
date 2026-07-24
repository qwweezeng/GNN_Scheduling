#!/usr/bin/env python3
"""Isolated TF1 inference worker for the retained DistGCN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf


def sparse_tuple(matrix):
    matrix = sp.coo_matrix(matrix)
    indices = np.column_stack((matrix.row, matrix.col)).astype(np.int64)
    return indices, matrix.data.astype(np.float32), np.asarray(matrix.shape, np.int64)


def normalized_laplacian_support(adjacency):
    adjacency = sp.coo_matrix(adjacency, dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = degree[positive] ** -0.5
    normalized = sp.diags(inverse).dot(adjacency).dot(sp.diags(inverse)).tocoo()
    laplacian = sp.eye(adjacency.shape[0], dtype=np.float32) - normalized
    return [sparse_tuple(sp.eye(adjacency.shape[0])), sparse_tuple(laplacian)]


def feature_tuple(weights, feature_size=32):
    weights = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    norm = float(np.linalg.norm(weights))
    if norm <= 0:
        dense = np.zeros((len(weights), feature_size), dtype=np.float32)
    else:
        dense = np.ones((len(weights), feature_size), dtype=np.float32)
        dense *= weights / norm
        row_sum = dense.sum(axis=1, keepdims=True)
        dense = np.divide(dense, row_sum, out=np.zeros_like(dense), where=row_sum != 0)
    return sparse_tuple(sp.coo_matrix(dense))


def local_greedy(adjacency, priorities):
    adjacency = np.asarray(adjacency, dtype=np.int8)
    priorities = np.asarray(priorities, dtype=np.float64).reshape(-1)
    remaining = set(range(len(priorities)))
    selected, blocked = set(), set()
    rounds = messages = 0
    while remaining:
        winners = set()
        for vertex in sorted(remaining):
            neighbors = set(np.flatnonzero(adjacency[vertex])).intersection(remaining)
            messages += len(neighbors)
            if not neighbors:
                winners.add(vertex)
                continue
            best = max(priorities[index] for index in neighbors)
            if priorities[vertex] > best:
                winners.add(vertex)
            elif priorities[vertex] == best:
                tied = min(index for index in neighbors if priorities[index] == best)
                if vertex < tied:
                    winners.add(vertex)
        if not winners:
            raise RuntimeError("DistGCN local greedy made no progress")
        newly_blocked = set()
        for winner in winners:
            newly_blocked.update(np.flatnonzero(adjacency[winner]).tolist())
        selected.update(winners)
        blocked.update(newly_blocked)
        remaining.difference_update(selected)
        remaining.difference_update(blocked)
        rounds += 1
    action = np.zeros(len(priorities), dtype=np.float32)
    action[list(selected)] = 1.0
    return action, {"contention_rounds": rounds, "directed_message_proxy": messages}


class DistGCNInference:
    def __init__(self, checkpoint):
        tf.disable_v2_behavior()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(
                str(checkpoint) + ".meta", clear_devices=True
            )
        self.session = tf.Session(graph=self.graph)
        self.saver.restore(self.session, str(checkpoint))
        self.output = self.graph.get_tensor_by_name("graphconvolution_3/AddN:0")
        first_weight = self.graph.get_tensor_by_name(
            "gcn_dqn/graphconvolution_1_vars/weights_0:0"
        )
        self.feature_size = int(first_weight.shape.as_list()[0])
        self.support = [
            tuple(self.graph.get_tensor_by_name(name) for name in names)
            for names in (
                ("Placeholder:0", "Placeholder_1:0", "Placeholder_2:0"),
                ("Placeholder_3:0", "Placeholder_4:0", "Placeholder_5:0"),
            )
        ]
        self.features = tuple(
            self.graph.get_tensor_by_name(name)
            for name in ("Placeholder_6:0", "Placeholder_7:0", "Placeholder_8:0")
        )
        self.num_features_nonzero = self.graph.get_tensor_by_name("Placeholder_11:0")

    def scores(self, adjacency, weights):
        supports = normalized_laplacian_support(adjacency)
        features = feature_tuple(weights, feature_size=self.feature_size)
        feed = {}
        for tensors, values in zip(self.support, supports):
            coordinates, sparse_values, dense_shape = values
            ordered = (dense_shape, sparse_values, coordinates)
            feed.update({tensor: value for tensor, value in zip(tensors, ordered)})
        coordinates, sparse_values, dense_shape = features
        ordered_features = (dense_shape, sparse_values, coordinates)
        feed.update({
            tensor: value for tensor, value in zip(self.features, ordered_features)
        })
        feed[self.num_features_nonzero] = np.asarray(features[1].shape, dtype=np.int32)
        return np.asarray(self.session.run(self.output, feed_dict=feed)).reshape(-1)

    def close(self):
        self.session.close()


def schedule(engine, adjacency, weights):
    adjacency = np.asarray(adjacency, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    positive = np.flatnonzero(weights > 0)
    action = np.zeros_like(weights, dtype=np.float32)
    scores = np.zeros_like(weights, dtype=np.float32)
    diagnostics = {"contention_rounds": 0, "directed_message_proxy": 0}
    if positive.size:
        reduced_adjacency = adjacency[np.ix_(positive, positive)]
        reduced_weights = weights[positive]
        reduced_scores = engine.scores(reduced_adjacency, reduced_weights)
        reduced_action, diagnostics = local_greedy(
            reduced_adjacency, reduced_scores * reduced_weights
        )
        action[positive] = reduced_action
        scores[positive] = reduced_scores
    return action, scores, diagnostics


def batch_main(args):
    payload = np.load(args.input, allow_pickle=False)
    adjacency = payload["adjacency"]
    weights = payload["weights"]
    if weights.ndim == 1:
        weights = weights[None, :]
    engine = DistGCNInference(args.checkpoint)
    actions, scores, diagnostics = [], [], []
    try:
        for row in weights:
            action, score, audit = schedule(engine, adjacency, row)
            actions.append(action)
            scores.append(score)
            diagnostics.append(audit)
    finally:
        engine.close()
    np.savez_compressed(
        args.output,
        actions=np.asarray(actions, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        diagnostics=np.asarray([json.dumps(item) for item in diagnostics]),
    )


def serve_main(args):
    engine = DistGCNInference(args.checkpoint)
    adjacency = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            command = request.get("command")
            if command == "initialize":
                adjacency = np.asarray(request["adjacency"], dtype=np.int8)
                response = {"status": "initialized", "num_nodes": len(adjacency)}
            elif command == "schedule":
                if adjacency is None:
                    raise RuntimeError("worker must be initialized first")
                action, scores, diagnostics = schedule(
                    engine, adjacency, np.asarray(request["weights"], dtype=np.float32)
                )
                response = {
                    "status": "ok",
                    "action": action.tolist(),
                    "scores": scores.tolist(),
                    "diagnostics": diagnostics,
                }
            elif command == "close":
                print(json.dumps({"status": "closed"}), flush=True)
                break
            else:
                raise ValueError(f"unknown command {command!r}")
            print(json.dumps(response), flush=True)
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve_main(args)
    elif args.input and args.output:
        batch_main(args)
    else:
        parser.error("use --serve or provide --input and --output")


if __name__ == "__main__":
    main()
