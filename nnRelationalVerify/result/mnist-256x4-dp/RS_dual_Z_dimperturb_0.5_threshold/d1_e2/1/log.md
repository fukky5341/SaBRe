## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00045437


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041957, -0.0015296, -0.0041957, -0.0015296, -0.0026660, 0.0026660)
1: (0.0048267, 0.0067084, 0.0048267, 0.0067084, -0.0018817, 0.0018817)
2: (0.0102239, 0.0152462, 0.0102239, 0.0152462, -0.0045025, 0.0045025)
3: (-0.0048983, -0.0026559, -0.0048983, -0.0026559, -0.0022423, 0.0022423)
4: (0.0044971, 0.0052444, 0.0044971, 0.0052444, -0.0007474, 0.0007474)
5: (-0.0024848, -0.0007956, -0.0024848, -0.0007956, -0.0016892, 0.0016892)
6: (-0.0060855, -0.0052771, -0.0060855, -0.0052771, -0.0008083, 0.0008083)
7: (-0.0033374, -0.0017624, -0.0033374, -0.0017624, -0.0015749, 0.0015749)
8: (-0.0045624, -0.0012361, -0.0045624, -0.0012361, -0.0033263, 0.0033263)
9: (1.0004160, 1.0010756, 1.0004160, 1.0010756, -0.0006596, 0.0006596)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.15 + 1.40 = 2.55 seconds
status: Status.ADV_EXAMPLE
