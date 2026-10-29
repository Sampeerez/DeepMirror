"""Model fixes applied once after download.

HyperSwap ships with fp16 weights cast to fp32 inside the graph; on CUDA that
path overflows to NaN on GPUs without tensor cores (GTX 16xx). Storing the
weights in fp32 and dropping the casts gives the same result as the CPU, and
it is faster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def to_float32(source: Path, target: Path) -> None:
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(str(source))
    graph = model.graph
    for index, tensor in enumerate(graph.initializer):
        if tensor.data_type == TensorProto.FLOAT16:
            converted = numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name)
            graph.initializer[index].CopyFrom(converted)
    for node in graph.node:
        if node.op_type == "Cast":
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
                    attribute.i = TensorProto.FLOAT
        elif node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.t.data_type == TensorProto.FLOAT16:
                    attribute.t.CopyFrom(numpy_helper.from_array(
                        numpy_helper.to_array(attribute.t).astype(np.float32), attribute.t.name))
    del graph.value_info[:]   # stale fp16 shape annotations; ONNX Runtime infers them again
    partial = target.with_suffix(".part")
    onnx.save(model, str(partial))
    partial.replace(target)
