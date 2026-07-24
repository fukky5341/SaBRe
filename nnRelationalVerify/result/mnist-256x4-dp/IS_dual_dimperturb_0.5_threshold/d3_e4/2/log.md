## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0181036


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0045337, -0.0002423, -0.0045337, -0.0002423, -0.0042914, 0.0042914)
1: (-0.0029912, 0.0032578, -0.0029912, 0.0032578, -0.0062490, 0.0062490)
2: (0.0076878, 0.0219473, 0.0076878, 0.0219473, -0.0142595, 0.0142595)
3: (-0.0028952, 0.0040947, -0.0028952, 0.0040947, -0.0069899, 0.0069899)
4: (0.9850852, 1.0126362, 0.9850852, 1.0126362, -0.0275509, 0.0275509)
5: (-0.0047194, 0.0071606, -0.0047194, 0.0071606, -0.0118800, 0.0118800)
6: (-0.0118588, -0.0057255, -0.0118588, -0.0057255, -0.0061333, 0.0061333)
7: (-0.0103161, -0.0007503, -0.0103161, -0.0007503, -0.0095658, 0.0095658)
8: (-0.0066530, -0.0026606, -0.0066530, -0.0026606, -0.0039924, 0.0039924)
9: (-0.0048513, 0.0205514, -0.0048513, 0.0205514, -0.0254027, 0.0254027)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.83 + 3.29 = 5.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0181733, upper bound: 0.0181733

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0171913, upper bound: 0.0177776
time: 1.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0177776, upper bound: 0.0177776
time: 2.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.12 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 4.12
Output dim: 4, lower bound: -0.0171913, upper bound: 0.0177776
IS_A2, status: Status.VERIFIED, split count: 1, time: 4.12
Output dim: 4, lower bound: -0.0177776, upper bound: 0.0177776

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 5.12 + 4.12 = 9.24 seconds
