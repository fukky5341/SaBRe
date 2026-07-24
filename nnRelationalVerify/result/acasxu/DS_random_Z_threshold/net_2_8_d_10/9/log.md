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
execution time: IAR + RelationalAnalysis = 1.06 + 1.59 = 2.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1420.6047579, upper bound: 1420.6047579

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.74 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9662912, upper bound: 1404.9662912
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9662912, upper bound: 1404.9662912
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 0, lower bound: -1404.9662912, upper bound: 1404.9662912
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 0, lower bound: -1404.9662912, upper bound: 1404.9662912

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1388.6665524, upper bound: 1388.6665524
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1388.6665524, upper bound: 1388.6665524
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9639527, upper bound: 1404.9639527
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9639527, upper bound: 1404.9639527
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.41 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -1388.6665524, upper bound: 1388.6665524
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -1388.6665524, upper bound: 1388.6665524
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1404.9639527, upper bound: 1404.9639527
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1404.9639527, upper bound: 1404.9639527

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1387.8216335, upper bound: 1387.8216335
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1387.8216335, upper bound: 1387.8216335
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9627248, upper bound: 1404.9627248
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9627248, upper bound: 1404.9627248
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1419.4510734, upper bound: 1419.4510734
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.69
Output dim: 0, lower bound: -1387.8216335, upper bound: 1387.8216335
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.69
Output dim: 0, lower bound: -1387.8216335, upper bound: 1387.8216335
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1404.9627248, upper bound: 1404.9627248
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -1404.9627248, upper bound: 1404.9627248

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2050044, upper bound: 1419.2050044
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2050044, upper bound: 1419.2050044
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150896, upper bound: 1417.9150896
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150896, upper bound: 1417.9150896
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.5968920, upper bound: 1404.5968920
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.5968920, upper bound: 1404.5968920
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9368836, upper bound: 1404.9368836
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9368836, upper bound: 1404.9368836
time: 0.46 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1419.2050044, upper bound: 1419.2050044
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1419.2050044, upper bound: 1419.2050044
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1417.9150896, upper bound: 1417.9150896
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1417.9150896, upper bound: 1417.9150896
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1407.0998741, upper bound: 1407.0998741
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1419.4510375, upper bound: 1419.4510375
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1404.5968920, upper bound: 1404.5968920
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1404.5968920, upper bound: 1404.5968920
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1404.9368836, upper bound: 1404.9368836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -1404.9368836, upper bound: 1404.9368836

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7927953, upper bound: 1418.7927953
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7927953, upper bound: 1418.7927953
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9109769, upper bound: 1417.9109769
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9109769, upper bound: 1417.9109769
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3657094, upper bound: 1419.3657094
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3657094, upper bound: 1419.3657094
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.5608654, upper bound: 1404.5608654
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.5608654, upper bound: 1404.5608654
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1387.2598120, upper bound: 1387.2598120
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1387.2598120, upper bound: 1387.2598120
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9057596, upper bound: 1404.9057596
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.9057596, upper bound: 1404.9057596
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.2534409, upper bound: 1404.2534409
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.2534409, upper bound: 1404.2534409
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1418.7927953, upper bound: 1418.7927953
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1418.7927953, upper bound: 1418.7927953
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9109769, upper bound: 1417.9109769
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9109769, upper bound: 1417.9109769
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1419.3657094, upper bound: 1419.3657094
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1419.3657094, upper bound: 1419.3657094
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1404.5608654, upper bound: 1404.5608654
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1404.5608654, upper bound: 1404.5608654
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.80
Output dim: 0, lower bound: -1387.2598120, upper bound: 1387.2598120
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.80
Output dim: 0, lower bound: -1387.2598120, upper bound: 1387.2598120
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1404.9057596, upper bound: 1404.9057596
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1404.9057596, upper bound: 1404.9057596
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1404.2534409, upper bound: 1404.2534409
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.80
Output dim: 0, lower bound: -1404.2534409, upper bound: 1404.2534409

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7923698, upper bound: 1418.7923698
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7923698, upper bound: 1418.7923698
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.6756824, upper bound: 1417.6756824
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.6756824, upper bound: 1417.6756824
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8919112, upper bound: 1417.8919112
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8919112, upper bound: 1417.8919112
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.5197963, upper bound: 1404.5197963
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.5197963, upper bound: 1404.5197963
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1395.1589405, upper bound: 1395.1589405
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1395.1589405, upper bound: 1395.1589405
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.8382286, upper bound: 1404.8382286
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.8382286, upper bound: 1404.8382286
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.2304885, upper bound: 1404.2304885
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.2304885, upper bound: 1404.2304885
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.8644752, upper bound: 1403.8644752
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.8644752, upper bound: 1403.8644752
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.1883766, upper bound: 1404.1883766
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.1883766, upper bound: 1404.1883766
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1418.7923698, upper bound: 1418.7923698
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1418.7923698, upper bound: 1418.7923698
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.6756824, upper bound: 1417.6756824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.6756824, upper bound: 1417.6756824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.9150644, upper bound: 1417.9150644
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.8406844, upper bound: 1417.8406844
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.8919112, upper bound: 1417.8919112
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1417.8919112, upper bound: 1417.8919112
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.5197963, upper bound: 1404.5197963
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.5197963, upper bound: 1404.5197963
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 0, lower bound: -1395.1589405, upper bound: 1395.1589405
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 0, lower bound: -1395.1589405, upper bound: 1395.1589405
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.8382286, upper bound: 1404.8382286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.8382286, upper bound: 1404.8382286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.2304885, upper bound: 1404.2304885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.2304885, upper bound: 1404.2304885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1403.8644752, upper bound: 1403.8644752
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1403.8644752, upper bound: 1403.8644752
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.1883766, upper bound: 1404.1883766
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 0, lower bound: -1404.1883766, upper bound: 1404.1883766

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0727540, upper bound: 1407.0727540
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0727540, upper bound: 1407.0727540
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7731744, upper bound: 1418.7731744
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7731744, upper bound: 1418.7731744
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7890905, upper bound: 1418.7890905
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7890905, upper bound: 1418.7890905
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7023159, upper bound: 1418.7023159
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7023159, upper bound: 1418.7023159
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.8011622, upper bound: 1416.8011622
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.8011622, upper bound: 1416.8011622
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.5933012, upper bound: 1417.5933012
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.5933012, upper bound: 1417.5933012
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5408690, upper bound: 1406.5408690
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5408690, upper bound: 1406.5408690
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530455, upper bound: 1406.5530455
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530455, upper bound: 1406.5530455
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.8637569, upper bound: 1418.8637569
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.8637569, upper bound: 1418.8637569
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.8128950, upper bound: 1416.8128950
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.8128950, upper bound: 1416.8128950
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.8778275, upper bound: 1416.8778275
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.8778275, upper bound: 1416.8778275
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1387.9697665, upper bound: 1387.9697665
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1387.9697665, upper bound: 1387.9697665
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
time: 0.40 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7474723, upper bound: 1406.7474723
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0814987, upper bound: 1407.0814987
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0727540, upper bound: 1407.0727540
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0727540, upper bound: 1407.0727540
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7731744, upper bound: 1418.7731744
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7731744, upper bound: 1418.7731744
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7736000, upper bound: 1418.7736000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7890905, upper bound: 1418.7890905
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7890905, upper bound: 1418.7890905
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7023159, upper bound: 1418.7023159
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.7023159, upper bound: 1418.7023159
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.8011622, upper bound: 1416.8011622
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.8011622, upper bound: 1416.8011622
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.5933012, upper bound: 1417.5933012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.5933012, upper bound: 1417.5933012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5408690, upper bound: 1406.5408690
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5408690, upper bound: 1406.5408690
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5530455, upper bound: 1406.5530455
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5530455, upper bound: 1406.5530455
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.8637569, upper bound: 1418.8637569
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1418.8637569, upper bound: 1418.8637569
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.8128950, upper bound: 1416.8128950
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.8128950, upper bound: 1416.8128950
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.8778275, upper bound: 1416.8778275
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1416.8778275, upper bound: 1416.8778275
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.09
Output dim: 0, lower bound: -1387.9697665, upper bound: 1387.9697665
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.09
Output dim: 0, lower bound: -1387.9697665, upper bound: 1387.9697665
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.09
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5408420, upper bound: 1406.5408420
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5408420, upper bound: 1406.5408420
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7772876, upper bound: 1416.7772876
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7772876, upper bound: 1416.7772876
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.6831205, upper bound: 1418.6831205
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.6831205, upper bound: 1418.6831205
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7789498, upper bound: 1416.7789498
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7789498, upper bound: 1416.7789498
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7703207, upper bound: 1418.7703207
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7703207, upper bound: 1418.7703207
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7997121, upper bound: 1416.7997121
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7997121, upper bound: 1416.7997121
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7042771, upper bound: 1416.7042771
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.7042771, upper bound: 1416.7042771
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705
1: -86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016
2: -142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504
3: -159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043
4: -122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
time: 0.46 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5408420, upper bound: 1406.5408420
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5408420, upper bound: 1406.5408420
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5411019, upper bound: 1406.5411019
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7376933, upper bound: 1406.7376933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5318737, upper bound: 1406.5318737
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.5413080, upper bound: 1406.5413080
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0733882, upper bound: 1407.0733882
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8797877, upper bound: 1406.8797877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8719214, upper bound: 1406.8719214
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8715696, upper bound: 1406.8715696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8794359, upper bound: 1406.8794359
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7772876, upper bound: 1416.7772876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7772876, upper bound: 1416.7772876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.6831205, upper bound: 1418.6831205
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.6831205, upper bound: 1418.6831205
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7789498, upper bound: 1416.7789498
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7789498, upper bound: 1416.7789498
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.7703207, upper bound: 1418.7703207
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.7703207, upper bound: 1418.7703207
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7997121, upper bound: 1416.7997121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7997121, upper bound: 1416.7997121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1418.6990367, upper bound: 1418.6990367
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7042771, upper bound: 1416.7042771
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1416.7042771, upper bound: 1416.7042771
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.8011622, upper bound: 1416.8011622
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.8011622, upper bound: 1416.8011622
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.5933012, upper bound: 1417.5933012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.5933012, upper bound: 1417.5933012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5408690, upper bound: 1406.5408690
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5408690, upper bound: 1406.5408690
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5530455, upper bound: 1406.5530455
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5530455, upper bound: 1406.5530455
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.8978517, upper bound: 1406.8978517
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0992430, upper bound: 1407.0992430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7588936, upper bound: 1406.7588936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.5530483, upper bound: 1406.5530483
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7491614, upper bound: 1406.7491614
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7468165, upper bound: 1406.7468165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0808645, upper bound: 1407.0808645
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.7370374, upper bound: 1406.7370374
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1418.8637569, upper bound: 1418.8637569
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1418.8637569, upper bound: 1418.8637569
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1419.3643083, upper bound: 1419.3643083
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1407.0911325, upper bound: 1407.0911325
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.9109513, upper bound: 1417.9109513
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.9132104, upper bound: 1416.9132104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.8128950, upper bound: 1416.8128950
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.8128950, upper bound: 1416.8128950
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8155687, upper bound: 1417.8155687
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1417.8371107, upper bound: 1417.8371107
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.8778275, upper bound: 1416.8778275
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1416.8778275, upper bound: 1416.8778275
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1406.8899045, upper bound: 1406.8899045
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.4489650, upper bound: 1404.4489650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.8363539, upper bound: 1403.8363539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1404.1651683, upper bound: 1404.1651683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -1403.7982869, upper bound: 1403.7982869

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.65 + 417.61 = 420.26 seconds
