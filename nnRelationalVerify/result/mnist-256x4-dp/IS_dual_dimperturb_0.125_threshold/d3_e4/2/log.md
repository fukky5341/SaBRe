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
Threshold: 0.00061831


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0039683, -0.0038238, -0.0039683, -0.0038238, -0.0000493, 0.0000493)
1: (0.0009203, 0.0017206, 0.0009203, 0.0017206, -0.0002727, 0.0002727)
2: (0.0111222, 0.0129100, 0.0111222, 0.0129100, -0.0006093, 0.0006093)
3: (0.0018940, 0.0026474, 0.0018940, 0.0026474, -0.0002568, 0.0002568)
4: (1.0040984, 1.0070213, 1.0040984, 1.0070213, -0.0009962, 0.0009962)
5: (0.0030387, 0.0036073, 0.0030387, 0.0036073, -0.0001938, 0.0001938)
6: (-0.0104374, -0.0096974, -0.0104374, -0.0096974, -0.0002522, 0.0002522)
7: (-0.0101347, -0.0100404, -0.0101347, -0.0100404, -0.0000322, 0.0000322)
8: (-0.0041540, -0.0036427, -0.0041540, -0.0036427, -0.0001742, 0.0001742)
9: (0.0000655, 0.0026249, 0.0000655, 0.0026249, -0.0008723, 0.0008723)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.88 + 1.31 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0009151, upper bound: 0.0009151

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007918, upper bound: 0.0007918
time: 0.48 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 3.19 + 1.19 = 4.38 seconds
