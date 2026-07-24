## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0032694, 1.0042883, 1.0032694, 1.0042883, -0.0005776, 0.0005776)
1: (-0.0004493, -0.0001954, -0.0004493, -0.0001954, -0.0001439, 0.0001439)
2: (-0.0090183, -0.0076730, -0.0090183, -0.0076730, -0.0007627, 0.0007627)
3: (0.0022193, 0.0028316, 0.0022193, 0.0028316, -0.0003471, 0.0003471)
4: (-0.0012176, -0.0009572, -0.0012176, -0.0009572, -0.0001476, 0.0001476)
5: (-0.0123832, -0.0106911, -0.0123832, -0.0106911, -0.0009593, 0.0009593)
6: (0.0042544, 0.0046838, 0.0042544, 0.0046838, -0.0002435, 0.0002435)
7: (0.0078697, 0.0089808, 0.0078697, 0.0089808, -0.0006299, 0.0006299)
8: (0.0045745, 0.0051588, 0.0045745, 0.0051588, -0.0003313, 0.0003313)
9: (-0.0078457, -0.0071682, -0.0078457, -0.0071682, -0.0003841, 0.0003841)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.36 = 2.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0003049, upper bound: 0.0003049

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002799, upper bound: 0.0002465
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002799, upper bound: 0.0002799
time: 0.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.26
Output dim: 0, lower bound: -0.0002799, upper bound: 0.0002465
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.26
Output dim: 0, lower bound: -0.0002799, upper bound: 0.0002799

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.64 + 1.26 = 3.91 seconds
