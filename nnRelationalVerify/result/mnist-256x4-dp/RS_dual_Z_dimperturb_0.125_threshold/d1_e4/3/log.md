## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.44e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0016894, -0.0015341, -0.0016894, -0.0015341, -0.0000616, 0.0000616)
1: (-0.0085978, -0.0082035, -0.0085978, -0.0082035, -0.0001563, 0.0001563)
2: (0.0296959, 0.0299405, 0.0296959, 0.0299405, -0.0000969, 0.0000969)
3: (0.0036559, 0.0041127, 0.0036559, 0.0041127, -0.0001810, 0.0001810)
4: (-0.0076384, -0.0072373, -0.0076384, -0.0072373, -0.0001589, 0.0001589)
5: (0.0108450, 0.0109969, 0.0108450, 0.0109969, -0.0000602, 0.0000602)
6: (0.0050280, 0.0056077, 0.0050280, 0.0056077, -0.0002297, 0.0002297)
7: (0.9815776, 0.9819833, 0.9815776, 0.9819833, -0.0001608, 0.0001608)
8: (-0.0063160, -0.0058810, -0.0063160, -0.0058810, -0.0001724, 0.0001724)
9: (-0.0011149, -0.0008276, -0.0011149, -0.0008276, -0.0001138, 0.0001138)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.45 + 1.25 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0001035, upper bound: 0.0001036

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0000982, upper bound: 0.0000984
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.ADV_EXAMPLE
time: 0.37 seconds

## RS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (rs) = 2.70 + 0.96 = 3.66 seconds
