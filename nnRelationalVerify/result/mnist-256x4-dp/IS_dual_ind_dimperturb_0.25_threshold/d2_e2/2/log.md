## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00149824981


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0000222, 0.0012291, -0.0000222, 0.0012291, -0.0010249, 0.0010249)
1: (0.9932763, 0.9959261, 0.9932763, 0.9959261, -0.0021913, 0.0021913)
2: (-0.0080170, -0.0070304, -0.0080170, -0.0070304, -0.0008006, 0.0008006)
3: (0.0026603, 0.0042258, 0.0026603, 0.0042258, -0.0012973, 0.0012973)
4: (0.0024655, 0.0045059, 0.0024655, 0.0045059, -0.0018980, 0.0018980)
5: (0.0033098, 0.0062749, 0.0033098, 0.0062749, -0.0023721, 0.0023721)
6: (-0.0019754, 0.0007665, -0.0019754, 0.0007665, -0.0023348, 0.0023348)
7: (-0.0079862, -0.0067178, -0.0079862, -0.0067178, -0.0009926, 0.0009926)
8: (0.0074887, 0.0081936, 0.0074887, 0.0081936, -0.0006525, 0.0006525)
9: (-0.0038395, -0.0020284, -0.0038395, -0.0020284, -0.0014780, 0.0014780)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.35 + 1.54 = 2.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0015354, upper bound: 0.0015354

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014434, upper bound: 0.0014906
time: 0.85 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014906, upper bound: 0.0014906
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.69
Output dim: 1, lower bound: -0.0014434, upper bound: 0.0014906
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.69
Output dim: 1, lower bound: -0.0014906, upper bound: 0.0014906

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.88 + 1.69 = 4.58 seconds
