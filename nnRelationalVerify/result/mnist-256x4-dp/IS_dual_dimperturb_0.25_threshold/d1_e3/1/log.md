## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018112


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040994, -0.0040839, -0.0040994, -0.0040839, -0.0000072, 0.0000072)
1: (-0.0061792, -0.0055970, -0.0061792, -0.0055970, -0.0002698, 0.0002698)
2: (0.9690481, 0.9697468, 0.9690481, 0.9697468, -0.0003237, 0.0003237)
3: (0.0180102, 0.0231636, 0.0180102, 0.0231636, -0.0023877, 0.0023877)
4: (-0.0024548, -0.0020628, -0.0024548, -0.0020628, -0.0001816, 0.0001816)
5: (0.0147894, 0.0151855, 0.0147894, 0.0151855, -0.0001835, 0.0001835)
6: (0.0045206, 0.0047132, 0.0045206, 0.0047132, -0.0000893, 0.0000893)
7: (-0.0137813, -0.0124458, -0.0137813, -0.0124458, -0.0006188, 0.0006188)
8: (0.0057957, 0.0068553, 0.0057957, 0.0068553, -0.0004909, 0.0004909)
9: (0.0081488, 0.0100545, 0.0081488, 0.0100545, -0.0008830, 0.0008830)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.35 = 2.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0002337, upper bound: 0.0002337

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002234, upper bound: 0.0002252
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002252, upper bound: 0.0002252
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.18 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 2, lower bound: -0.0002234, upper bound: 0.0002252
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 2, lower bound: -0.0002252, upper bound: 0.0002252

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040989, -0.0040839, -0.0040993, -0.0040839, -0.0000066, 0.0000068
1: -0.0061601, -0.0055990, -0.0061733, -0.0055970, -0.0002482, 0.0002535
2: 0.9690712, 0.9697444, 0.9690552, 0.9697468, -0.0002978, 0.0003042
3: 0.0181798, 0.0231458, 0.0180628, 0.0231633, -0.0021968, 0.0022438
4: -0.0024534, -0.0020757, -0.0024547, -0.0020668, -0.0001707, 0.0001671
5: 0.0147907, 0.0151725, 0.0147894, 0.0151815, -0.0001725, 0.0001689
6: 0.0045269, 0.0047126, 0.0045225, 0.0047132, -0.0000821, 0.0000839
7: -0.0137767, -0.0124897, -0.0137812, -0.0124594, -0.0005815, 0.0005693
8: 0.0057994, 0.0068204, 0.0057958, 0.0068445, -0.0004613, 0.0004517
9: 0.0081553, 0.0099918, 0.0081489, 0.0100350, -0.0008298, 0.0008124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.ADV_EXAMPLE
time: 0.54 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.ADV_EXAMPLE
time: 0.51 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.63 + 3.48 = 6.12 seconds
