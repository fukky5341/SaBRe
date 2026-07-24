## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.005605665000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647)
1: (-0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365)
2: (-0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179)
3: (-0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761)
4: (-0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.87 + 0.69 = 3.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0086241, upper bound: 0.0086241

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057176, upper bound: 0.0056874
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056874, upper bound: 0.0057176
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0057176, upper bound: 0.0056874
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0056874, upper bound: 0.0057176

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057169, upper bound: 0.0056677
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056677, upper bound: 0.0056785
time: 0.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056785, upper bound: 0.0056677
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056677, upper bound: 0.0057169
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.52 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -0.0057169, upper bound: 0.0056677
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -0.0056677, upper bound: 0.0056785
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -0.0056785, upper bound: 0.0056677
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -0.0056677, upper bound: 0.0057169

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057169, upper bound: 0.0056674
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057079, upper bound: 0.0056667
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056670, upper bound: 0.0056781
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056674, upper bound: 0.0056647
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056647, upper bound: 0.0056674
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056781, upper bound: 0.0056670
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056667, upper bound: 0.0057079
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056674, upper bound: 0.0057169
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0057169, upper bound: 0.0056674
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0057079, upper bound: 0.0056667
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0056670, upper bound: 0.0056781
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0056674, upper bound: 0.0056647
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0056647, upper bound: 0.0056674
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0056781, upper bound: 0.0056670
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0056667, upper bound: 0.0057079
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 0, lower bound: -0.0056674, upper bound: 0.0057169

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057043, upper bound: 0.0056532
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056697, upper bound: 0.0056523
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056953, upper bound: 0.0056514
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056650, upper bound: 0.0056515
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056520, upper bound: 0.0056602
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056524, upper bound: 0.0056628
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056523, upper bound: 0.0056437
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056532, upper bound: 0.0056482
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056482, upper bound: 0.0056532
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056437, upper bound: 0.0056523
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056628, upper bound: 0.0056524
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056602, upper bound: 0.0056520
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056515, upper bound: 0.0056650
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056514, upper bound: 0.0056953
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056523, upper bound: 0.0056697
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056532, upper bound: 0.0057043
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0057043, upper bound: 0.0056532
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056697, upper bound: 0.0056523
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056953, upper bound: 0.0056514
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056650, upper bound: 0.0056515
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056520, upper bound: 0.0056602
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056524, upper bound: 0.0056628
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056523, upper bound: 0.0056437
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056532, upper bound: 0.0056482
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056482, upper bound: 0.0056532
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056437, upper bound: 0.0056523
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056628, upper bound: 0.0056524
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056602, upper bound: 0.0056520
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056515, upper bound: 0.0056650
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056514, upper bound: 0.0056953
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056523, upper bound: 0.0056697
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -0.0056532, upper bound: 0.0057043

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055989, upper bound: 0.0055660
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056193, upper bound: 0.0055528
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055758, upper bound: 0.0055663
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055781, upper bound: 0.0055593
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055604, upper bound: 0.0055639
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056102, upper bound: 0.0055618
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055644
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055770, upper bound: 0.0055624
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055640, upper bound: 0.0055747
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055655, upper bound: 0.0055519
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055634, upper bound: 0.0055780
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055650, upper bound: 0.0055585
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055594, upper bound: 0.0055519
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055663, upper bound: 0.0055519
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055519
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055660, upper bound: 0.0055564
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055564, upper bound: 0.0055660
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055519
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055663
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055594
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055585, upper bound: 0.0055650
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055780, upper bound: 0.0055634
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055655
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055747, upper bound: 0.0055640
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055624, upper bound: 0.0055770
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055644, upper bound: 0.0055519
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055618, upper bound: 0.0056102
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055639, upper bound: 0.0055604
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055593, upper bound: 0.0055781
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055663, upper bound: 0.0055758
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055528, upper bound: 0.0056193
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055660, upper bound: 0.0055989
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055989, upper bound: 0.0055660
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0056193, upper bound: 0.0055528
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055758, upper bound: 0.0055663
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055781, upper bound: 0.0055593
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055604, upper bound: 0.0055639
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0056102, upper bound: 0.0055618
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055644
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055770, upper bound: 0.0055624
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055640, upper bound: 0.0055747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055655, upper bound: 0.0055519
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055634, upper bound: 0.0055780
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055650, upper bound: 0.0055585
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055594, upper bound: 0.0055519
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055663, upper bound: 0.0055519
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055519
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055660, upper bound: 0.0055564
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055564, upper bound: 0.0055660
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055519
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055663
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055594
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055585, upper bound: 0.0055650
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055780, upper bound: 0.0055634
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055519, upper bound: 0.0055655
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055747, upper bound: 0.0055640
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055624, upper bound: 0.0055770
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055644, upper bound: 0.0055519
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055618, upper bound: 0.0056102
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055639, upper bound: 0.0055604
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055593, upper bound: 0.0055781
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055663, upper bound: 0.0055758
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055528, upper bound: 0.0056193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.61
Output dim: 0, lower bound: -0.0055660, upper bound: 0.0055989

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054075, upper bound: 0.0053938
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0053938
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054034, upper bound: 0.0053938
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053993, upper bound: 0.0053938
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0053993
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0054034
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0053938
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0054075
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0054075, upper bound: 0.0053938
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0053938
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0054034, upper bound: 0.0053938
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0053993, upper bound: 0.0053938
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0053993
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0054034
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0053938
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 0, lower bound: -0.0053938, upper bound: 0.0054075

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.56 + 126.18 = 129.75 seconds
