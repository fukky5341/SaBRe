## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.020861730000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693)
1: (-0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900)
2: (0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209)
3: (-0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987)
4: (0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823)
5: (-0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120)
6: (-0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657)
7: (-0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077)
8: (-0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964)
9: (-0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.13 + 3.54 = 5.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0231797, upper bound: 0.0231797

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0176071, upper bound: 0.0176071
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0176071, upper bound: 0.0176071
time: 1.09 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.47 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 2.47
Output dim: 4, lower bound: -0.0176071, upper bound: 0.0176071
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 2.47
Output dim: 4, lower bound: -0.0176071, upper bound: 0.0176071

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.67 + 2.47 = 8.14 seconds
