import tensorrt as trt

engine_path = "4PhaseFinal.engine"  # Adjust to your filename
logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

print("\n=== TENSORRT ENGINE INSPECTION ===")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)

    print(f"\nTensor {i}: {name}")
    print(f"  Direction: {'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'}")
    print(f"  Shape:     {list(shape)}")
    print(f"  DataType:  {dtype}")
print("\n==================================")