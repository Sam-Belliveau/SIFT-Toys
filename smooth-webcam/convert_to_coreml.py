"""
Convert Kornia SIFT to Core ML

This script attempts to convert the Kornia SIFTFeature model to Core ML
for faster inference on Apple Silicon.
"""

import torch
import kornia.feature as KF
import coremltools as ct

# Parameters
NUM_FEATURES = 64
INPUT_HEIGHT = 720
INPUT_WIDTH = 1280

print("Creating Kornia SIFT model...")
model = KF.SIFTFeature(num_features=NUM_FEATURES)
model.eval()

# Create example input
example_input = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH)

print("Tracing model with TorchScript...")
try:
    # Try tracing
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    print("Tracing succeeded!")
except Exception as e:
    print(f"Tracing failed: {e}")
    print("\nTrying torch.jit.script instead...")
    try:
        scripted = torch.jit.script(model)
        traced = scripted
        print("Scripting succeeded!")
    except Exception as e2:
        print(f"Scripting also failed: {e2}")
        print("\nKornia SIFT uses dynamic control flow that doesn't convert well.")
        print("Consider using a simpler feature detector or OpenCV SIFT.")
        exit(1)

print("\nConverting to Core ML...")
try:
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="image",
                shape=(1, 1, INPUT_HEIGHT, INPUT_WIDTH),
            )
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,  # Use GPU + Neural Engine
    )

    output_path = "sift.mlpackage"
    mlmodel.save(output_path)
    print(f"\nSaved Core ML model to: {output_path}")

except Exception as e:
    print(f"Core ML conversion failed: {e}")
    print("\nThe model likely uses operations not supported by Core ML.")
