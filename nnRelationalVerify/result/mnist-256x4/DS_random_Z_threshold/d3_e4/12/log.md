## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.6947652919999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548)
1: (-0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906)
2: (-0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625)
3: (-1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869)
4: (-1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024)
5: (-0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151)
6: (-0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975)
7: (-0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587)
8: (-1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157)
9: (-1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.88 + 2.87 = 3.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6697041, upper bound: 1.6697041
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6697041, upper bound: 1.6697041
time: 1.02 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.06 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 2.06
Output dim: 5, lower bound: -1.6697041, upper bound: 1.6697041
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 2.06
Output dim: 5, lower bound: -1.6697041, upper bound: 1.6697041

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.75 + 2.06 = 5.81 seconds
