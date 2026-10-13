"""Small TensorRT 10 inference wrapper using PyTorch tensors for CUDA buffers."""

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTEngine:
    """Load a serialized engine and run float32 NCHW batches on the NVIDIA GPU.

    Host-to-device and device-to-host copies are included in each ``run`` call.
    This makes the timing representative of calling the engine with CPU arrays.
    """

    def __init__(self, engine_path: Path):
        engine_path = Path(engine_path)
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires an available NVIDIA CUDA GPU.")
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.logger = LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

        inputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        outputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Expected one input and one output tensor, found {inputs} and {outputs}"
            )

        self.input_name = inputs[0]
        self.output_name = outputs[0]
        profile_min, profile_opt, profile_max = self.engine.get_tensor_profile_shape(
            self.input_name, 0
        )
        self.min_batch_size = int(profile_min[0])
        self.opt_batch_size = int(profile_opt[0])
        self.max_batch_size = int(profile_max[0])
        self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        if self.input_dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
            raise ValueError(f"Unsupported classifier input dtype: {self.input_dtype}")
        self.stream = torch.cuda.Stream()

    def run(self, input_array: np.ndarray) -> np.ndarray:
        if input_array.ndim != 4 or input_array.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected NCHW input (N, 3, 224, 224), got {input_array.shape}")
        input_array = np.ascontiguousarray(input_array, dtype=self.input_dtype)
        input_shape = tuple(input_array.shape)
        if not self.min_batch_size <= input_shape[0] <= self.max_batch_size:
            raise ValueError(
                f"Batch size {input_shape[0]} is outside this engine profile range "
                f"[{self.min_batch_size}, {self.max_batch_size}] "
                f"(optimized for {self.opt_batch_size})."
            )

        with torch.cuda.stream(self.stream):
            input_gpu = torch.from_numpy(input_array).to("cuda")
            if not self.context.set_input_shape(self.input_name, input_shape):
                raise ValueError(f"TensorRT rejected input shape: {input_shape}")

            output_shape = tuple(self.context.get_tensor_shape(self.output_name))
            if any(dimension < 0 for dimension in output_shape):
                raise RuntimeError(f"TensorRT output shape is unresolved: {output_shape}")

            output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
            output_gpu = torch.empty(
                output_shape,
                dtype=torch.from_numpy(np.empty((), dtype=output_dtype)).dtype,
                device="cuda",
            )
            if not self.context.set_tensor_address(self.input_name, input_gpu.data_ptr()):
                raise RuntimeError(f"Could not set TensorRT input address: {self.input_name}")
            if not self.context.set_tensor_address(self.output_name, output_gpu.data_ptr()):
                raise RuntimeError(f"Could not set TensorRT output address: {self.output_name}")
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed.")

        self.stream.synchronize()
        return output_gpu.cpu().numpy()
