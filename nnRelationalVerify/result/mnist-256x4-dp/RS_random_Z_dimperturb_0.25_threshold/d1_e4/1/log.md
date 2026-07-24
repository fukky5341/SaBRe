## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00048692


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0032173, 0.0044399, 0.0032173, 0.0044399, -0.0006390, 0.0006390)
1: (0.0017871, 0.0019637, 0.0017871, 0.0019637, -0.0000923, 0.0000923)
2: (0.0119052, 0.0125811, 0.0119052, 0.0125811, -0.0003533, 0.0003533)
3: (-0.0023676, -0.0016685, -0.0023676, -0.0016685, -0.0003654, 0.0003654)
4: (-0.0022307, -0.0014739, -0.0022307, -0.0014739, -0.0003956, 0.0003956)
5: (0.0055082, 0.0062244, 0.0055082, 0.0062244, -0.0003743, 0.0003743)
6: (-0.0004455, 0.0023962, -0.0004455, 0.0023962, -0.0014852, 0.0014852)
7: (-0.0058201, -0.0019500, -0.0058201, -0.0019500, -0.0020227, 0.0020227)
8: (0.9851140, 0.9878402, 0.9851140, 0.9878402, -0.0014249, 0.0014249)
9: (-0.0048494, -0.0023748, -0.0048494, -0.0023748, -0.0012934, 0.0012934)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.28 = 2.70 seconds
status: Status.ADV_EXAMPLE
