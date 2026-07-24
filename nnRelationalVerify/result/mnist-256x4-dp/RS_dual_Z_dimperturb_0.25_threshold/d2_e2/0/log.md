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
Threshold: 0.00011788


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041712, -0.0041575, -0.0041712, -0.0041575, -0.0000094, 0.0000094)
1: (-0.0088682, -0.0083534, -0.0088682, -0.0083534, -0.0003520, 0.0003520)
2: (0.9658213, 0.9664389, 0.9658213, 0.9664389, -0.0004224, 0.0004224)
3: (-0.0057910, -0.0012346, -0.0057910, -0.0012346, -0.0031153, 0.0031153)
4: (-0.0005991, -0.0002526, -0.0005991, -0.0002526, -0.0002369, 0.0002369)
5: (0.0166648, 0.0170151, 0.0166648, 0.0170151, -0.0002395, 0.0002395)
6: (0.0036307, 0.0038010, 0.0036307, 0.0038010, -0.0001165, 0.0001165)
7: (-0.0074583, -0.0062775, -0.0074583, -0.0062775, -0.0008074, 0.0008074)
8: (0.0108121, 0.0117489, 0.0108121, 0.0117489, -0.0006405, 0.0006405)
9: (0.0171712, 0.0188561, 0.0171712, 0.0188561, -0.0011520, 0.0011520)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.21 = 2.39 seconds
status: Status.ADV_EXAMPLE
