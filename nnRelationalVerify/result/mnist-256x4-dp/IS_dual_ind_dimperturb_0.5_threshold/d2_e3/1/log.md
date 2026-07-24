## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00027335


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0066002, 0.0072373, 0.0066002, 0.0072373, -0.0003792, 0.0003792)
1: (0.0008660, 0.0021002, 0.0008660, 0.0021002, -0.0007345, 0.0007345)
2: (-0.0000117, 0.0099429, -0.0000117, 0.0099429, -0.0059241, 0.0059241)
3: (-0.0034006, -0.0025115, -0.0034006, -0.0025115, -0.0005291, 0.0005291)
4: (0.0048370, 0.0091507, 0.0048370, 0.0091507, -0.0025671, 0.0025671)
5: (-0.0018857, -0.0012418, -0.0018857, -0.0012418, -0.0003832, 0.0003832)
6: (0.9925306, 0.9937117, 0.9925306, 0.9937117, -0.0007029, 0.0007029)
7: (-0.0046271, 0.0031815, -0.0046271, 0.0031815, -0.0046470, 0.0046470)
8: (-0.0004613, 0.0019851, -0.0004613, 0.0019851, -0.0014559, 0.0014559)
9: (-0.0112911, -0.0064085, -0.0112911, -0.0064085, -0.0029057, 0.0029057)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.78 = 3.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0005399, upper bound: 0.0005398

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.66 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 3.09 + 1.42 = 4.52 seconds
