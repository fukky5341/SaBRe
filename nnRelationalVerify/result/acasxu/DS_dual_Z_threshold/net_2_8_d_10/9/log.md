## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 1399.2956865315


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705)
1: (-86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016)
2: (-142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504)
3: (-159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043)
4: (-122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.70 + 1.71 = 4.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1420.6047579, upper bound: 1420.6047579

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.5857585, upper bound: 1405.5857585
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.5857585, upper bound: 1405.5857585
time: 0.50 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -1405.5857585, upper bound: 1405.5857585
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -1405.5857585, upper bound: 1405.5857585

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275
time: 0.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -1405.5498275, upper bound: 1405.5498275

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
time: 0.46 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.66 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.66
Output dim: 0, lower bound: -1396.7095455, upper bound: 1396.7095455

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.41 + 23.28 = 27.70 seconds
