## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00603328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0031479, -0.0019014, -0.0031479, -0.0019014, -0.0010449, 0.0010449)
1: (0.0225254, 0.0290563, 0.0225254, 0.0290563, -0.0032706, 0.0032706)
2: (0.0224180, 0.0268957, 0.0224180, 0.0268957, -0.0024657, 0.0024657)
3: (0.0099457, 0.0150774, 0.0099457, 0.0150774, -0.0035337, 0.0035337)
4: (-0.0154397, -0.0100500, -0.0154397, -0.0100500, -0.0038003, 0.0038003)
5: (0.0170644, 0.0232807, 0.0170644, 0.0232807, -0.0042679, 0.0042679)
6: (0.0079529, 0.0128538, 0.0079529, 0.0128538, -0.0035022, 0.0035022)
7: (-0.0201152, -0.0151109, -0.0201152, -0.0151109, -0.0033499, 0.0033499)
8: (0.0120004, 0.0168918, 0.0120004, 0.0168918, -0.0032825, 0.0032825)
9: (0.9110597, 0.9350525, 0.9110597, 0.9350525, -0.0155145, 0.0155145)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.31 = 2.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0084355, upper bound: 0.0084355

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.42 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.63 + 1.05 = 3.68 seconds
