## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.029087099999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0074664, 0.0208159, -0.0074664, 0.0208159, -0.0282824, 0.0282824)
1: (-0.0114187, 0.0159950, -0.0114187, 0.0159950, -0.0274137, 0.0274137)
2: (-0.0526190, 0.0148103, -0.0526190, 0.0148103, -0.0674294, 0.0674294)
3: (-0.0272276, 0.0312855, -0.0272276, 0.0312855, -0.0585130, 0.0585130)
4: (0.0078912, 0.0179667, 0.0078912, 0.0179667, -0.0100755, 0.0100755)
5: (-0.0395915, 0.0445230, -0.0395915, 0.0445230, -0.0841145, 0.0841145)
6: (0.9707767, 1.0273691, 0.9707767, 1.0273691, -0.0565925, 0.0565925)
7: (0.0009016, 0.0308351, 0.0009016, 0.0308351, -0.0293946, 0.0293946)
8: (-0.0081205, 0.0168595, -0.0081205, 0.0168595, -0.0249800, 0.0249800)
9: (-0.0501618, 0.0064838, -0.0501618, 0.0064838, -0.0566456, 0.0566456)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.81 + 3.00 = 4.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0306180, upper bound: 0.0306180

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0260057, upper bound: 0.0260057
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0260057, upper bound: 0.0260057
time: 1.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.02 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 3.02
Output dim: 6, lower bound: -0.0260057, upper bound: 0.0260057
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 3.02
Output dim: 6, lower bound: -0.0260057, upper bound: 0.0260057

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.80 + 3.02 = 7.82 seconds
