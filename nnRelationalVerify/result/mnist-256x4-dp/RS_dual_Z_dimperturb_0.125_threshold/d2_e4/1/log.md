## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00086954


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0061207, 0.0082566, 0.0061207, 0.0082566, -0.0010444, 0.0010444)
1: (0.0019958, 0.0040305, 0.0019958, 0.0040305, -0.0009973, 0.0009973)
2: (-0.0202253, -0.0151330, -0.0202253, -0.0151330, -0.0024473, 0.0024473)
3: (-0.0012437, 0.0031752, -0.0012437, 0.0031752, -0.0021523, 0.0021523)
4: (0.0153815, 0.0157573, 0.0153815, 0.0157573, -0.0002013, 0.0002013)
5: (-0.0030008, 0.0032221, -0.0030008, 0.0032221, -0.0030381, 0.0030381)
6: (0.9954307, 0.9996236, 0.9954307, 0.9996236, -0.0020409, 0.0020409)
7: (0.0150760, 0.0169277, 0.0150760, 0.0169277, -0.0008836, 0.0008836)
8: (0.0038802, 0.0057667, 0.0038802, 0.0057667, -0.0009296, 0.0009296)
9: (-0.0229487, -0.0186708, -0.0229487, -0.0186708, -0.0020685, 0.0020685)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.78 + 1.37 = 3.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0012674, upper bound: 0.0012674

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008031, upper bound: 0.0008031
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008031, upper bound: 0.0008031
time: 0.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.06 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.06
Output dim: 6, lower bound: -0.0008031, upper bound: 0.0008031
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.06
Output dim: 6, lower bound: -0.0008031, upper bound: 0.0008031

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.14 + 1.06 = 4.20 seconds
