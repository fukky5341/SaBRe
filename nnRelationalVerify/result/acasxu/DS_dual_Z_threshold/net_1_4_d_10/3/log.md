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
execution time: IAR + RelationalAnalysis = 0.59 + 0.75 = 1.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.44
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.44
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739207
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 0.96 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 0.96
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739207
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.96
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 0.96
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.96
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733262, upper bound: 0.0751000
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.17 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 0.93 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.93
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.93
Output dim: 0, lower bound: -0.0733262, upper bound: 0.0751000
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.93
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.93
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733790, upper bound: 0.0749010
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765
time: 0.20 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.01 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0733790, upper bound: 0.0749010
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.01
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0743066
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745305
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0743207
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745404
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0737937
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746782
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746147
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0730543
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746810
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746282
time: 0.20 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.00 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0743066
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745305
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0743207
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745404
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0737937
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746782
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746147
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0730543
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746810
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.00
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746282

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732466
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732599
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732726
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733124
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727600, upper bound: 0.0732194
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732606
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731978
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733096
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732996
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733817
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0730703
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733095
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731612
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727826, upper bound: 0.0732634
time: 0.20 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.24 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732466
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732599
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732726
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733124
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727600, upper bound: 0.0732194
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732606
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731978
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733096
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732996
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733817
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0730703
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0733095
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731612
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.24
Output dim: 0, lower bound: -0.0727826, upper bound: 0.0732634

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.34 + 28.41 = 29.75 seconds
