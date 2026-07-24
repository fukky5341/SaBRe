## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00046665


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0026603, -0.0021248, -0.0026603, -0.0021248, -0.0002564, 0.0002564)
1: (-0.0110616, -0.0097025, -0.0110616, -0.0097025, -0.0006506, 0.0006506)
2: (0.0281674, 0.0290105, 0.0281674, 0.0290105, -0.0004037, 0.0004037)
3: (0.0053925, 0.0069669, 0.0053925, 0.0069669, -0.0007537, 0.0007537)
4: (-0.0101445, -0.0087621, -0.0101445, -0.0087621, -0.0006618, 0.0006618)
5: (0.0098957, 0.0104193, 0.0098957, 0.0104193, -0.0002507, 0.0002507)
6: (0.0072319, 0.0092300, 0.0072319, 0.0092300, -0.0009566, 0.0009566)
7: (0.9831198, 0.9845180, 0.9831198, 0.9845180, -0.0006694, 0.0006694)
8: (-0.0046625, -0.0031634, -0.0046625, -0.0031634, -0.0007177, 0.0007177)
9: (-0.0029100, -0.0019198, -0.0029100, -0.0019198, -0.0004741, 0.0004741)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 1.38 = 3.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0004806, upper bound: 0.0004806

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004467, upper bound: 0.0004665
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004666, upper bound: 0.0004666
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.19 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.19
Output dim: 7, lower bound: -0.0004467, upper bound: 0.0004665
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.19
Output dim: 7, lower bound: -0.0004666, upper bound: 0.0004666

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.12 + 1.19 = 4.32 seconds
