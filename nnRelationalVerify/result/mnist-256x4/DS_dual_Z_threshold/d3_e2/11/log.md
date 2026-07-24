## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01217727


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0008509, 0.0102428, 0.0008509, 0.0102428, -0.0093892, 0.0093892)
1: (0.0010717, 0.0027631, 0.0010717, 0.0027631, -0.0016915, 0.0016915)
2: (0.0088459, 0.0144952, 0.0088459, 0.0144952, -0.0056493, 0.0056493)
3: (-0.0055316, -0.0000963, -0.0055316, -0.0000963, -0.0053872, 0.0053872)
4: (-0.0043037, 0.0019513, -0.0043037, 0.0019513, -0.0062550, 0.0062550)
5: (0.0013203, 0.0076106, 0.0013203, 0.0076106, -0.0062903, 0.0062903)
6: (-0.0164160, 0.0078963, -0.0164160, 0.0078963, -0.0243123, 0.0243123)
7: (-0.0133108, 0.0168555, -0.0133108, 0.0168555, -0.0301663, 0.0301663)
8: (0.9797018, 1.0001787, 0.9797018, 1.0001787, -0.0202530, 0.0202530)
9: (-0.0160495, 0.0036146, -0.0160495, 0.0036146, -0.0195672, 0.0195672)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.97 + 2.91 = 4.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0143262, upper bound: 0.0143262

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133763, upper bound: 0.0133856
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133856, upper bound: 0.0133763
time: 1.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.30 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.30
Output dim: 8, lower bound: -0.0133763, upper bound: 0.0133856
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.30
Output dim: 8, lower bound: -0.0133856, upper bound: 0.0133763

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0008509, 0.0102428, 0.0008509, 0.0102428, -0.0092594, 0.0093065
1: 0.0010717, 0.0027631, 0.0010717, 0.0027631, -0.0016915, 0.0016915
2: 0.0088459, 0.0144952, 0.0088459, 0.0144952, -0.0056311, 0.0056051
3: -0.0055316, -0.0000963, -0.0055316, -0.0000963, -0.0053425, 0.0053155
4: -0.0043037, 0.0019513, -0.0043037, 0.0019513, -0.0061927, 0.0062219
5: 0.0013203, 0.0076106, 0.0013203, 0.0076106, -0.0062903, 0.0062903
6: -0.0164160, 0.0078963, -0.0164160, 0.0078963, -0.0243123, 0.0243123
7: -0.0133108, 0.0168555, -0.0133108, 0.0168555, -0.0299104, 0.0300597
8: 0.9797018, 1.0001787, 0.9797018, 1.0001787, -0.0199643, 0.0200695
9: -0.0160495, 0.0036146, -0.0160495, 0.0036146, -0.0194139, 0.0193184

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249
time: 1.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0008509, 0.0102428, 0.0008509, 0.0102428, -0.0093892, 0.0092594
1: 0.0010717, 0.0027631, 0.0010717, 0.0027631, -0.0016915, 0.0016915
2: 0.0088459, 0.0144952, 0.0088459, 0.0144952, -0.0056051, 0.0056493
3: -0.0055316, -0.0000963, -0.0055316, -0.0000963, -0.0053155, 0.0053872
4: -0.0043037, 0.0019513, -0.0043037, 0.0019513, -0.0062550, 0.0061927
5: 0.0013203, 0.0076106, 0.0013203, 0.0076106, -0.0062903, 0.0062903
6: -0.0164160, 0.0078963, -0.0164160, 0.0078963, -0.0243123, 0.0243123
7: -0.0133108, 0.0168555, -0.0133108, 0.0168555, -0.0301663, 0.0299104
8: 0.9797018, 1.0001787, 0.9797018, 1.0001787, -0.0202530, 0.0199643
9: -0.0160495, 0.0036146, -0.0160495, 0.0036146, -0.0193184, 0.0195672

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249
time: 1.13 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.15 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.0111249, upper bound: 0.0111249

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.88 + 11.67 = 16.55 seconds
