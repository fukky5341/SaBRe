## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00217782


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0025802, -0.0012373, -0.0025802, -0.0012373, -0.0009106, 0.0009106)
1: (-0.0026716, 0.0006279, -0.0026716, 0.0006279, -0.0023549, 0.0023549)
2: (0.0044019, 0.0072410, 0.0044019, 0.0072410, -0.0019602, 0.0019602)
3: (-0.0042597, -0.0038583, -0.0042597, -0.0038583, -0.0002626, 0.0002626)
4: (0.0036105, 0.0062241, 0.0036105, 0.0062241, -0.0015194, 0.0015194)
5: (-0.0013949, 0.0012190, -0.0013949, 0.0012190, -0.0015992, 0.0015992)
6: (-0.0058505, -0.0043317, -0.0058505, -0.0043317, -0.0008161, 0.0008161)
7: (-0.0000908, 0.0024095, -0.0000908, 0.0024095, -0.0014801, 0.0014801)
8: (-0.0005167, -0.0001517, -0.0005167, -0.0001517, -0.0002471, 0.0002471)
9: (1.0026668, 1.0087506, 1.0026668, 1.0087506, -0.0040409, 0.0040409)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.36 = 2.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0022197, upper bound: 0.0022197

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021223, upper bound: 0.0018568
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021256, upper bound: 0.0021256
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.20 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.20
Output dim: 9, lower bound: -0.0021223, upper bound: 0.0018568
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.20
Output dim: 9, lower bound: -0.0021256, upper bound: 0.0021256

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.86 + 1.20 = 4.06 seconds
