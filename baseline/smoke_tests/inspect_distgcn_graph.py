#!/usr/bin/env python3
"""List the feed and inference tensors retained in a DistGCN meta graph."""

import argparse
from pathlib import Path

import tensorflow.compat.v1 as tf


DEFAULT = Path(__file__).resolve().parents[1] / (
    "distgcn-main/model/"
    "result_DQNBA_deep_ld1_c32_l3_cheb1_diver1_mwis_dqn/model.ckpt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT))
    args = parser.parse_args()
    tf.disable_v2_behavior()
    graph = tf.Graph()
    with graph.as_default():
        tf.train.import_meta_graph(args.checkpoint + ".meta", clear_devices=True)
    print("PLACEHOLDERS")
    def shapes(op):
        output = []
        for tensor in op.outputs:
            try:
                output.append(tensor.shape.as_list())
            except ValueError:
                output.append("unknown")
        return output
    for op in graph.get_operations():
        if op.type in {"Placeholder", "PlaceholderWithDefault"}:
            print(
                op.name,
                op.type,
                shapes(op),
                [tensor.dtype.name for tensor in op.outputs],
                [[consumer.name for consumer in tensor.consumers()[:3]] for tensor in op.outputs],
            )
    print("FINAL_LAYER")
    for op in graph.get_operations():
        if op.name.startswith("graphconvolution_3/") and not op.name.startswith(
            "graphconvolution_3/graphconvolution_3/"
        ):
            print(op.name, op.type, shapes(op))
    print("INFERENCE_CANDIDATES")
    for op in graph.get_operations():
        lowered = op.name.lower()
        if (
            any(key in lowered for key in ("softmax", "pred", "output"))
            or op.type in {
                "ArgMax", "Softmax", "Sigmoid", "Identity", "MatMul",
                "SparseTensorDenseMatMul", "Mul",
            }
        ):
            print(op.name, op.type, shapes(op))


if __name__ == "__main__":
    main()
