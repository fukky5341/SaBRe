## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00046428


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041768, -0.0041461, -0.0041768, -0.0041461, -0.0000276, 0.0000276)
1: (-0.0090768, -0.0079255, -0.0090768, -0.0079255, -0.0010338, 0.0010338)
2: (0.9655709, 0.9669525, 0.9655709, 0.9669525, -0.0012407, 0.0012407)
3: (-0.0076371, 0.0025530, -0.0076371, 0.0025530, -0.0091509, 0.0091509)
4: (-0.0008872, -0.0001122, -0.0008872, -0.0001122, -0.0006960, 0.0006960)
5: (0.0163737, 0.0171570, 0.0163737, 0.0171570, -0.0007034, 0.0007034)
6: (0.0035616, 0.0039426, 0.0035616, 0.0039426, -0.0003421, 0.0003421)
7: (-0.0084399, -0.0057990, -0.0084399, -0.0057990, -0.0023715, 0.0023715)
8: (0.0100333, 0.0121285, 0.0100333, 0.0121285, -0.0018815, 0.0018815)
9: (0.0157706, 0.0195388, 0.0157706, 0.0195388, -0.0033840, 0.0033840)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.12 + 1.40 = 2.52 seconds
status: Status.ADV_EXAMPLE
