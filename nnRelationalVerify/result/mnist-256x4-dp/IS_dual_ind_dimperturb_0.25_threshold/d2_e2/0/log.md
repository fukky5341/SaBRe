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
0: (-0.0041717, -0.0041575, -0.0041717, -0.0041575, -0.0000093, 0.0000093)
1: (-0.0088869, -0.0083524, -0.0088869, -0.0083524, -0.0003488, 0.0003488)
2: (0.9657987, 0.9664403, 0.9657987, 0.9664403, -0.0004185, 0.0004185)
3: (-0.0059566, -0.0012254, -0.0059566, -0.0012254, -0.0030870, 0.0030870)
4: (-0.0005998, -0.0002400, -0.0005998, -0.0002400, -0.0002348, 0.0002348)
5: (0.0166641, 0.0170278, 0.0166641, 0.0170278, -0.0002373, 0.0002373)
6: (0.0036245, 0.0038014, 0.0036245, 0.0038014, -0.0001154, 0.0001154)
7: (-0.0074607, -0.0062345, -0.0074607, -0.0062345, -0.0008000, 0.0008000)
8: (0.0108102, 0.0117829, 0.0108102, 0.0117829, -0.0006347, 0.0006347)
9: (0.0171678, 0.0189174, 0.0171678, 0.0189174, -0.0011416, 0.0011416)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.22 = 2.44 seconds
status: Status.ADV_EXAMPLE
