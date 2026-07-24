## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.030823649999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650)
1: (-0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866)
2: (-0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139)
3: (-0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888)
4: (-0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017)
5: (0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867)
6: (-0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650)
7: (-0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367)
8: (-0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854)
9: (-0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.76 + 2.28 = 4.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295850, upper bound: 0.0295850
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295850, upper bound: 0.0295850
time: 1.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.10 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 3.10
Output dim: 5, lower bound: -0.0295850, upper bound: 0.0295850
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 3.10
Output dim: 5, lower bound: -0.0295850, upper bound: 0.0295850

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.04 + 3.10 = 7.14 seconds
