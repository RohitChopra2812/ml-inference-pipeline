import pytest
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "/Users/I760158/ml_inference_env/resnet18.onnx"
LABELS_PATH = "/Users/I760158/ml_inference_env/imagenet_classes.txt"
INPUT_SHAPE = (1, 3, 224, 224)


@pytest.fixture(scope="module")
def session():
    """Create ONNX Runtime session once, reuse across all tests."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(MODEL_PATH, sess_options=opts)


@pytest.fixture(scope="module")
def labels():
    """Load ImageNet class labels."""
    with open(LABELS_PATH) as f:
        return [line.strip() for line in f.readlines()]


def run_inference(session, input_data):
    """Helper: run inference and return output array."""
    return session.run(["output"], {"input": input_data})[0]


# --- Test 1: Model loads correctly ---
def test_model_loads():
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(MODEL_PATH, sess_options=opts)
    assert sess is not None


# --- Test 2: Output shape is correct [1, 1000] ---
def test_output_shape(session):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    output = run_inference(session, dummy)
    assert output.shape == (1, 1000), f"Expected (1, 1000), got {output.shape}"


# --- Test 3: Output is 1000 classes (ImageNet) ---
def test_output_classes(session):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    output = run_inference(session, dummy)
    assert output.shape[1] == 1000


# --- Test 4: Deterministic — same input always gives same output ---
def test_deterministic(session):
    fixed_input = np.ones(INPUT_SHAPE, dtype=np.float32) * 0.5
    out1 = run_inference(session, fixed_input)
    out2 = run_inference(session, fixed_input)
    np.testing.assert_array_equal(out1, out2)


# --- Test 5: Different inputs give different outputs ---
def test_different_inputs_differ(session):
    input1 = np.zeros(INPUT_SHAPE, dtype=np.float32)
    input2 = np.ones(INPUT_SHAPE, dtype=np.float32)
    out1 = run_inference(session, input1)
    out2 = run_inference(session, input2)
    assert not np.array_equal(out1, out2)


# --- Test 6: Top predicted class is consistent across runs ---
def test_top_class_stable(session):
    fixed_input = np.ones(INPUT_SHAPE, dtype=np.float32) * 0.5
    results = [np.argmax(run_inference(session, fixed_input)) for _ in range(5)]
    assert len(set(results)) == 1, "Top class changed across runs — model is non-deterministic"


# --- Test 7: Latency is within acceptable range ---
def test_latency(session):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    times = []
    for _ in range(20):
        start = time.time()
        run_inference(session, dummy)
        times.append((time.time() - start) * 1000)
    avg_ms = np.mean(times)
    print(f"\nAvg latency: {avg_ms:.2f} ms")
    assert avg_ms < 200, f"Inference too slow: {avg_ms:.2f} ms (threshold: 200ms)"


# --- Test 8: Labels file has exactly 1000 entries ---
def test_labels_count(labels):
    assert len(labels) == 1000, f"Expected 1000 labels, got {len(labels)}"


# --- Test 9: Top-1 prediction maps to a valid label ---
def test_prediction_has_valid_label(session, labels):
    fixed_input = np.ones(INPUT_SHAPE, dtype=np.float32) * 0.5
    output = run_inference(session, fixed_input)
    top_class = int(np.argmax(output))
    assert 0 <= top_class < 1000
    assert len(labels[top_class]) > 0
    print(f"\nTop prediction: [{top_class}] {labels[top_class]}")
