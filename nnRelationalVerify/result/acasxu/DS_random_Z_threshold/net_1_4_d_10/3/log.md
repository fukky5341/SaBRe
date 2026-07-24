## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.07398656999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144)
1: (-0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303)
2: (-0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905)
3: (-0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251)
4: (-0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.92 + 0.79 = 1.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743
time: 0.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.23 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737033, upper bound: 0.0751743
time: 0.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.18 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.18
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.18
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.18
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.18
Output dim: 0, lower bound: -0.0737033, upper bound: 0.0751743

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.27 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.27
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.27
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.27
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.27
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0737937
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747913
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731285, upper bound: 0.0748469
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733882, upper bound: 0.0730980
time: 0.24 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.37 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0737937
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747913
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0731285, upper bound: 0.0748469
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.37
Output dim: 0, lower bound: -0.0733882, upper bound: 0.0730980

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0741860
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0746662
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0747038
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0746974
time: 0.23 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.67 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0741860
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0746662
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0747038
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.67
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0746974

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0729562
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0740713
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728240, upper bound: 0.0731604
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728240, upper bound: 0.0732425
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0729566
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0744005
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0728664
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0745337
time: 0.23 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.37 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0729562
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0740713
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0728240, upper bound: 0.0731604
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0728240, upper bound: 0.0732425
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0729566
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0744005
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0728664
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.37
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0745337

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0740610
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0739058
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0724809, upper bound: 0.0743292
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724809, upper bound: 0.0738919
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0725687
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0742086
time: 0.23 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.54 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.54
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0740610
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.54
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0739058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.54
Output dim: 0, lower bound: -0.0724809, upper bound: 0.0743292
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.54
Output dim: 0, lower bound: -0.0724809, upper bound: 0.0738919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.54
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0725687
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.54
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0742086

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0737829
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0723264
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0738909
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0742515
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0738061
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0740921
time: 0.22 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.62 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.62
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0737829
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.62
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0723264
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.62
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0738909
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.62
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0742515
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.62
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0738061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.62
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0740921

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 48

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0722366
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0741032
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0740259
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737392
time: 0.24 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.71 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.71
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0722366
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.71
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0741032
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.71
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0740259
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.71
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737392

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 48

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0721416
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723404
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 7

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0721416
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723914
time: 0.24 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 1.33 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.33
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0721416
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.33
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723404
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.33
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0721416
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.33
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723914

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.71 + 37.78 = 39.50 seconds
