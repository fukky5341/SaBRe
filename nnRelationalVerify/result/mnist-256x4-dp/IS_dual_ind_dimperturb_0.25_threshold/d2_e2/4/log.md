## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0004263


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9877189, 0.9891146, 0.9877189, 0.9891146, -0.0011806, 0.0011806)
1: (-0.0043241, -0.0039763, -0.0043241, -0.0039763, -0.0002942, 0.0002942)
2: (0.0110183, 0.0128613, 0.0110183, 0.0128613, -0.0015590, 0.0015590)
3: (-0.0071270, -0.0062882, -0.0071270, -0.0062882, -0.0007096, 0.0007096)
4: (0.0026605, 0.0030172, 0.0026605, 0.0030172, -0.0003017, 0.0003017)
5: (0.0128176, 0.0151356, 0.0128176, 0.0151356, -0.0019608, 0.0019608)
6: (-0.0023008, -0.0017124, -0.0023008, -0.0017124, -0.0004977, 0.0004977)
7: (-0.0090904, -0.0075681, -0.0090904, -0.0075681, -0.0012876, 0.0012876)
8: (-0.0043447, -0.0035442, -0.0043447, -0.0035442, -0.0006772, 0.0006772)
9: (0.0022458, 0.0031740, 0.0022458, 0.0031740, -0.0007852, 0.0007852)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 1.49 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0007624, upper bound: 0.0007624

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.46 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.93 + 1.11 = 4.04 seconds
