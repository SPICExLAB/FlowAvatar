"""Export the FlowAvatar LSTM model to ONNX for Unity Sentis (on-device mode).

Unity Sentis has no GRU operator, so the on-device configuration uses the
LSTM variant. The export uses a fixed input shape [1, seq_len, 90] and
opset 15 as recommended by the Unity Sentis documentation.

Run from the repository root:

    python -m streaming.export_onnx --model_path checkpoints/LSTM/baseline.pt
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowavatar.models import load_checkpoint
from flowavatar.config import paths

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    sys.exit("ONNX export requires extra packages: pip install onnx onnxruntime")

try:
    import onnxsim
except ImportError:
    onnxsim = None
    print("onnx-simplifier not installed (pip install onnx-simplifier); "
          "the simplification pass will be skipped.")


SEQ_LEN = 40


class OnnxExportWrapper(torch.nn.Module):
    """ONNX-specific wrapper with fixed input/output shape (last frame only)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sparse_input):
        """
        Forward pass that returns only the last frame for ONNX export

        Args:
            sparse_input: Input features [1, 40, 90] fixed shape

        Returns:
            last_frame_pose: Pose parameters [1, 132]
            last_frame_shapes: Shape parameters [1, 10]
        """
        # Get outputs from the model
        outputs = self.model(sparse_input)

        # Extract only the last frame
        last_frame_pose = outputs["pose_params"][:, -1]  # [B, 132]
        last_frame_shapes = outputs["betas"][:, -1]  # [B, 10]

        return last_frame_pose, last_frame_shapes


def convert_to_onnx_for_sentis(model, output_path, opset_version=15):
    """
    Convert PyTorch model to ONNX format optimized for Unity Sentis on CPU

    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        opset_version: ONNX opset version (default: 15 as recommended by Unity Sentis docs)
    """
    print("Converting model to ONNX format optimized for Unity Sentis (CPU)...")
    print(f"Using ONNX opset version {opset_version} as recommended by Unity Sentis documentation")

    # Fixed input shape for on-device inference
    input_shape = (1, SEQ_LEN, 90)

    # Create example input
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, dtype=torch.float32, device=device)

    # Export model to ONNX with settings optimized for Sentis
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pred_pose', 'pred_shapes'],
        # No dynamic axes - fixed input size for better performance
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL
    )

    print(f"Model exported to {output_path}")

    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")

    return output_path


def simplify_onnx_model(input_path, output_path=None):
    """
    Simplify ONNX model using ONNX Simplifier

    Args:
        input_path: Input ONNX model path
        output_path: Output simplified ONNX model path

    Returns:
        Path to simplified model
    """
    if onnxsim is None:
        print("onnx-simplifier not available, skipping simplification.")
        return input_path

    if output_path is None:
        # Create output path
        input_dir = os.path.dirname(input_path)
        input_file = os.path.basename(input_path)
        output_path = os.path.join(input_dir, input_file.replace('.onnx', '_simplified.onnx'))

    print("Simplifying ONNX model with onnx-simplifier...")

    # Load the model
    model = onnx.load(input_path)

    # Simplify with fixed input shape
    try:
        model_simplified, check_ok = onnxsim.simplify(
            model,
            check_n=3,
            perform_optimization=True,
            skip_fuse_bn=False,
            input_shapes={"input": [1, SEQ_LEN, 90]}
        )

        if check_ok:
            print("Model simplified successfully")
            # Save the simplified model
            onnx.save(model_simplified, output_path)
            print(f"Simplified model saved to {output_path}")
        else:
            print("Model simplification failed! Using original model.")
            output_path = input_path

    except Exception as e:
        print(f"Error during simplification: {e}")
        print("Using original model.")
        output_path = input_path

    return output_path


def test_onnx_inference(model_path):
    """
    Test ONNX model inference speed on CPU

    Args:
        model_path: Path to ONNX model
    """
    print(f"Testing ONNX inference speed for {os.path.basename(model_path)}...")

    # Ensure we're using CPU provider only for Unity Sentis compatibility
    providers = ['CPUExecutionProvider']

    # Session options for CPU optimization
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4  # Adjust based on target device
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_cpu_mem_arena = True

    try:
        session = ort.InferenceSession(model_path, session_options, providers=providers)

        # Fixed input shape (1, 40, 90)
        dummy_input = np.random.randn(1, SEQ_LEN, 90).astype(np.float32)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        # Warm-up runs
        for _ in range(10):
            _ = session.run(output_names, {input_name: dummy_input})

        # Benchmark
        start_time = time.time()
        n_runs = 50  # More runs for accurate timing
        for _ in range(n_runs):
            _ = session.run(output_names, {input_name: dummy_input})
        onnx_time_ms = (time.time() - start_time) * 1000 / n_runs

        print(f"ONNX model average inference time (CPU): {onnx_time_ms:.2f} ms")
        return onnx_time_ms
    except Exception as e:
        print(f"Error during inference testing: {e}")
        return float('inf')


def validate_for_sentis(model_path):
    """
    Validates that the ONNX model is compatible with Unity Sentis

    Args:
        model_path: Path to the ONNX model

    Returns:
        bool: Whether the model is likely compatible with Unity Sentis
    """
    print(f"Checking Unity Sentis compatibility for {os.path.basename(model_path)}...")

    # Load the model
    model = onnx.load(model_path)

    # Operators unsupported by Sentis, per Unity documentation
    unsupported_ops = [
        'BitShift', 'ConcatFromSequence', 'ConvInteger', 'DequantizeLinear', 'Det',
        'DynamicQuantizeLinear', 'EyeLike', 'If', 'GRU', 'Loop', 'LpPool',
        'MatMulInteger', 'MaxUnpool', 'MeanVarianceNormalization',
        'NegativeLogLikelihoodLoss', 'Optional', 'OptionalGetElement',
        'OptionalHasElement', 'QLinearConv', 'QLinearMatMul', 'QuantizeLinear',
        'ReverseSequence', 'RNN', 'Scan', 'SequenceAt', 'SequenceConstruct',
        'SequenceEmpty', 'SequenceErase', 'SequenceInsert', 'SequenceLength',
        'SoftmaxCrossEntropyLoss', 'SplitToSequence', 'StringNormalizer',
        'TfIdfVectorizer', 'Unique'
    ]

    # LSTM parameters with limited support in Sentis
    lstm_unsupported_params = {
        'activation_alpha': 'LSTM alpha parameter',
        'activation_beta': 'LSTM beta parameter',
        'clip': 'LSTM clip parameter',
        'direction': 'Bidirectional LSTM',
        'layout': 'LSTM layout',
        'output_sequence': 'LSTM output sequence'
    }

    # Check model opset
    opset_version = model.opset_import[0].version
    if opset_version != 15:
        print(f"Warning: Model uses opset {opset_version}, but Unity Sentis recommends opset 15")

    # Check operators
    found_issues = []
    for node in model.graph.node:
        # Check if operator is unsupported
        if node.op_type in unsupported_ops:
            found_issues.append(f"Unsupported operator: {node.op_type}")

        # Check specific constraints on LSTM
        if node.op_type == 'LSTM':
            for attr in node.attribute:
                if attr.name in lstm_unsupported_params:
                    if attr.name == 'direction' and attr.s.decode() != 'forward':
                        found_issues.append(f"LSTM with direction '{attr.s.decode()}' might have limited support")
                    else:
                        found_issues.append(f"LSTM using {lstm_unsupported_params[attr.name]} which might have limited support")

    # Count LSTM nodes
    lstm_count = sum(1 for node in model.graph.node if node.op_type == 'LSTM')

    # Check tensor types (Sentis CPU supports float32 for LSTM)
    for tensor in model.graph.initializer:
        if tensor.data_type != 1:  # 1 is FLOAT
            found_issues.append(f"Non-float tensor: {tensor.name} with type {tensor.data_type}")

    # Report results
    if not found_issues:
        print("Model appears compatible with Unity Sentis (CPU backend)")
        if lstm_count > 0:
            print(f"  - Found {lstm_count} LSTM nodes (supported in CPU backend with float32)")
        print(f"  - Model uses opset version {opset_version} (Unity Sentis recommends opset 15)")
        return True
    else:
        print("Model may have compatibility issues with Unity Sentis:")
        for issue in found_issues:
            print(f"  - {issue}")
        return False


def export_for_sentis(model_path):
    """
    Full pipeline to export the LSTM model for Unity Sentis

    Args:
        model_path: Path to the PyTorch model checkpoint
    """
    # Create output directory
    converted_dir = paths.onnx_dir
    os.makedirs(converted_dir, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for PyTorch model: {device}")

    # Load the model with LSTM configuration
    print(f"Loading LSTM model from {model_path}")
    model = load_checkpoint(model_path, model_type='lstm')
    model = model.to(device)
    model.eval()

    # Create ONNX export wrapper
    onnx_model = OnnxExportWrapper(model)

    # Perform PyTorch model benchmark
    print("\nBenchmarking PyTorch model...")
    dummy_input = torch.randn(1, SEQ_LEN, 90, device=device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = onnx_model(dummy_input)

    # Benchmark
    start_time = time.time()
    n_runs = 50
    for _ in range(n_runs):
        with torch.no_grad():
            _ = onnx_model(dummy_input)
    torch_time_ms = (time.time() - start_time) * 1000 / n_runs
    print(f"PyTorch model average inference time: {torch_time_ms:.2f} ms")

    # Export and optionally simplify
    onnx_path = os.path.join(converted_dir, "flowavatar_sentis_lstm.onnx")
    convert_to_onnx_for_sentis(onnx_model, onnx_path, opset_version=15)
    plain_time_ms = test_onnx_inference(onnx_path)

    simplified_path = simplify_onnx_model(onnx_path)
    best_path, best_time = onnx_path, plain_time_ms
    if simplified_path != onnx_path:
        simplified_time_ms = test_onnx_inference(simplified_path)
        if simplified_time_ms < plain_time_ms:
            best_path, best_time = simplified_path, simplified_time_ms

    print("\n=============================================")
    print("EXPORT RESULTS")
    print("=============================================")
    print(f"PyTorch model: {torch_time_ms:.2f} ms")
    print(f"Best ONNX model: {best_path} ({best_time:.2f} ms, "
          f"{torch_time_ms / best_time:.2f}x speedup)")

    # Validate the best model for Sentis
    is_compatible = validate_for_sentis(best_path)

    final_path = os.path.join(converted_dir, "flowavatar_sentis_final.onnx")
    import shutil
    shutil.copy2(best_path, final_path)
    print(f"\nFinal model copied to: {final_path}")

    return {
        'path': final_path,
        'time': best_time,
        'pytorch_time': torch_time_ms,
        'compatible': is_compatible
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the FlowAvatar LSTM model to ONNX for Unity Sentis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LSTM model checkpoint")

    args = parser.parse_args()

    print("Exporting LSTM model for Unity Sentis (CPU backend)...")
    print(f"Using fixed input shape [1, {SEQ_LEN}, 90]")

    results = export_for_sentis(args.model_path)

    if results:
        print("\n=============================================")
        print("EXPORT COMPLETE")
        print("=============================================")
        print(f"Final model saved to: {results['path']}")
        print(f"PyTorch inference time: {results['pytorch_time']:.2f} ms")
        print(f"ONNX inference time: {results['time']:.2f} ms")
