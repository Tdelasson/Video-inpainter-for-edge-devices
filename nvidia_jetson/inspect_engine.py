import tensorrt as trt

engine_path = "4PhaseFinal.engine"  # Adjust to your filename

logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, 'rb') as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

print("\n--- EXACT TENSORRT ENGINE EXPECTATIONS (TRT 10+ API) ---")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    mode = engine.get_tensor_mode(name)

    role = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
    print(f"Tensor {i} ({role}): Name='{name}' | Shape={list(shape)} | Type={dtype}")
print("---------------------------------------------------------\n")