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
Threshold: 0.00365364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0032530, -0.0005097, -0.0032530, -0.0005097, -0.0022662, 0.0022662)
1: (-0.0125656, -0.0056040, -0.0125656, -0.0056040, -0.0057509, 0.0057509)
2: (0.0272343, 0.0315533, 0.0272343, 0.0315533, -0.0035679, 0.0035679)
3: (0.0006445, 0.0087092, 0.0006445, 0.0087092, -0.0066621, 0.0066621)
4: (-0.0116743, -0.0045932, -0.0116743, -0.0045932, -0.0058496, 0.0058496)
5: (0.0093163, 0.0119984, 0.0093163, 0.0119984, -0.0022157, 0.0022157)
6: (0.0012061, 0.0114412, 0.0012061, 0.0114412, -0.0084551, 0.0084551)
7: (0.9789032, 0.9860653, 0.9789032, 0.9860653, -0.0059165, 0.0059165)
8: (-0.0091833, -0.0015044, -0.0091833, -0.0015044, -0.0063434, 0.0063434)
9: (-0.0040058, 0.0010665, -0.0040058, 0.0010665, -0.0041902, 0.0041902)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 2.14 = 3.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0037577, upper bound: 0.0037577

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036098, upper bound: 0.0036446
time: 1.15 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036446, upper bound: 0.0036446
time: 1.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.39 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 2.39
Output dim: 7, lower bound: -0.0036098, upper bound: 0.0036446
IS_A2, status: Status.VERIFIED, split count: 1, time: 2.39
Output dim: 7, lower bound: -0.0036446, upper bound: 0.0036446

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.72 + 2.39 = 6.11 seconds
