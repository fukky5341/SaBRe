## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.851544332


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3877775, 0.4621432, -0.3877775, 0.4621432, -0.8499205, 0.8499205)
1: (-0.5284067, 2.0988278, -0.5284067, 2.0988278, -2.6272345, 2.6272345)
2: (-0.3809817, 0.5539122, -0.3809817, 0.5539122, -0.9348937, 0.9348938)
3: (-0.3085160, 0.3773541, -0.3085160, 0.3773541, -0.6858702, 0.6858701)
4: (-0.3952212, 0.4911242, -0.3952212, 0.4911242, -0.8863454, 0.8863454)
5: (-0.4354007, 0.5080089, -0.4354007, 0.5080089, -0.9434096, 0.9434096)
6: (-0.4014111, 0.4879104, -0.4014111, 0.4879104, -0.8893216, 0.8893216)
7: (-0.3190002, 0.8904487, -0.3190002, 0.8904487, -1.2094488, 1.2094488)
8: (-0.2782317, 0.7783343, -0.2782317, 0.7783343, -1.0565660, 1.0565660)
9: (-0.4433337, 0.5317172, -0.4433337, 0.5317172, -0.9750509, 0.9750510)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.18 + 6.01 = 8.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -1.8702468, upper bound: 1.8702468

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8645610, upper bound: 1.8370714
time: 2.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8374717, upper bound: 1.8374717
time: 4.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.45 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.45
Output dim: 1, lower bound: -1.8645610, upper bound: 1.8370714
NS_A2, status: Status.VERIFIED, split count: 1, time: 7.45
Output dim: 1, lower bound: -1.8374717, upper bound: 1.8374717

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.3713488, 0.4455409, -0.3877775, 0.4621432, -0.8334920, 0.8333184
1: -0.4980903, 2.0674815, -0.5284067, 2.0988278, -2.5969181, 2.5958879
2: -0.3667552, 0.5410592, -0.3809817, 0.5539122, -0.9206673, 0.9220406
3: -0.2972560, 0.3632430, -0.3085160, 0.3773541, -0.6746099, 0.6717591
4: -0.3816388, 0.4720399, -0.3952212, 0.4911242, -0.8727629, 0.8672611
5: -0.4205605, 0.4892426, -0.4354007, 0.5080089, -0.9285694, 0.9246432
6: -0.3849232, 0.4685152, -0.4014111, 0.4879104, -0.8728336, 0.8699263
7: -0.3050115, 0.8712559, -0.3190002, 0.8904487, -1.1954598, 1.1902560
8: -0.2648045, 0.7554450, -0.2782317, 0.7783343, -1.0431387, 1.0336766
9: -0.4260443, 0.5153025, -0.4433337, 0.5317172, -0.9577615, 0.9586361

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8370714, upper bound: 1.8370714
time: 4.44 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8370714, upper bound: 1.8370714
time: 4.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 10.71 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 10.71
Output dim: 1, lower bound: -1.8370714, upper bound: 1.8370714
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 10.71
Output dim: 1, lower bound: -1.8370714, upper bound: 1.8370714

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 8.19 + 18.16 = 26.35 seconds
