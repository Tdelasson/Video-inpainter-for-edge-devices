import tensorrt as trt

engine_path = "4PhaseFinal.engine"  # or video_inpainter_dynamic.engine

logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, 'rb') as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

print("\n--- EXACT TENSORRT ENGINE EXPECTATIONS ---")
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    dtype = engine.get_binding_dtype(i)
    shape = engine.get_binding_shape(i)
    is_input = engine.binding_is_input(i)
    role = "INPUT" if is_input else "OUTPUT"
    print(f"Binding {i} ({role}): Name='{name}' | Shape={list(shape)} | Type={dtype}")
print("-------------------------------------------\n")