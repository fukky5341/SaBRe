## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.00109514


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557)
1: (-0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170)
2: (-0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386)
3: (-0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066)
4: (-0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.94 + 0.63 = 1.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012884, upper bound: 0.0012884

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
time: 0.20 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.39 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.39
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.39
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0011801
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0011770
time: 0.15 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011770, upper bound: 0.0011813
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011801, upper bound: 0.0011813
time: 0.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0011801
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0011770
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -0.0011770, upper bound: 0.0011813
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -0.0011801, upper bound: 0.0011813

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010425, upper bound: 0.0010455
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010446, upper bound: 0.0010455
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010693
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010693
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010693, upper bound: 0.0010714
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010693, upper bound: 0.0010714
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010720, upper bound: 0.0010712
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010720, upper bound: 0.0010712
time: 0.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.28 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010425, upper bound: 0.0010455
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010446, upper bound: 0.0010455
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010693
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010693
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010693, upper bound: 0.0010714
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010693, upper bound: 0.0010714
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010720, upper bound: 0.0010712
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.28
Output dim: 0, lower bound: -0.0010720, upper bound: 0.0010712

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.57 + 8.37 = 9.95 seconds
