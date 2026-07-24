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
execution time: IAR + RelationalAnalysis = 1.03 + 0.54 = 1.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0086241, upper bound: 0.0086241

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0086205, upper bound: 0.0085432
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085432, upper bound: 0.0086205
time: 0.15 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.32
Output dim: 0, lower bound: -0.0086205, upper bound: 0.0085432
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.32
Output dim: 0, lower bound: -0.0085432, upper bound: 0.0086205

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057132, upper bound: 0.0056850
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056794, upper bound: 0.0057165
time: 0.15 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078751, upper bound: 0.0081841
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076634, upper bound: 0.0082730
time: 0.15 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.26 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0057132, upper bound: 0.0056850
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0056794, upper bound: 0.0057165
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0078751, upper bound: 0.0081841
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0076634, upper bound: 0.0082730

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057132, upper bound: 0.0056847
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057065, upper bound: 0.0056739
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055869, upper bound: 0.0056315
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055945, upper bound: 0.0056082
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078644, upper bound: 0.0079648
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077735, upper bound: 0.0081809
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060744, upper bound: 0.0061945
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060744, upper bound: 0.0062360
time: 0.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.25 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0057132, upper bound: 0.0056847
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0057065, upper bound: 0.0056739
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0055869, upper bound: 0.0056315
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0055945, upper bound: 0.0056082
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0078644, upper bound: 0.0079648
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0077735, upper bound: 0.0081809
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0060744, upper bound: 0.0061945
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0060744, upper bound: 0.0062360

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057125, upper bound: 0.0056649
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056644, upper bound: 0.0056757
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056911, upper bound: 0.0056621
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056863, upper bound: 0.0056629
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055868, upper bound: 0.0056234
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055846, upper bound: 0.0056314
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055877, upper bound: 0.0055816
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055941, upper bound: 0.0056082
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078312, upper bound: 0.0077847
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077309, upper bound: 0.0079510
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074285, upper bound: 0.0080900
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077364, upper bound: 0.0078952
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056643, upper bound: 0.0056473
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056498, upper bound: 0.0056497
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054797, upper bound: 0.0054797
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054797, upper bound: 0.0054806
time: 0.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0057125, upper bound: 0.0056649
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0056644, upper bound: 0.0056757
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0056911, upper bound: 0.0056621
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0056863, upper bound: 0.0056629
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0055868, upper bound: 0.0056234
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0055846, upper bound: 0.0056314
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0055877, upper bound: 0.0055816
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0055941, upper bound: 0.0056082
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0078312, upper bound: 0.0077847
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0077309, upper bound: 0.0079510
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0074285, upper bound: 0.0080900
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0077364, upper bound: 0.0078952
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0056643, upper bound: 0.0056473
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0056498, upper bound: 0.0056497
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0054797, upper bound: 0.0054797
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.50
Output dim: 0, lower bound: -0.0054797, upper bound: 0.0054806

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056984, upper bound: 0.0056480
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056908, upper bound: 0.0056535
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055773, upper bound: 0.0055908
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055783, upper bound: 0.0055675
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054903, upper bound: 0.0054851
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054854, upper bound: 0.0054851
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055268, upper bound: 0.0055200
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055265, upper bound: 0.0055200
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054074, upper bound: 0.0054102
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054074, upper bound: 0.0054136
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055685, upper bound: 0.0055832
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0055633, upper bound: 0.0056188
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055792, upper bound: 0.0055799
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055641, upper bound: 0.0055965
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0070916, upper bound: 0.0069085
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069323, upper bound: 0.0070748
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0070916, upper bound: 0.0069043
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069508, upper bound: 0.0071108
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067563, upper bound: 0.0067563
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067833, upper bound: 0.0071714
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054020, upper bound: 0.0053998
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053998, upper bound: 0.0053998
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056434, upper bound: 0.0056448
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056638, upper bound: 0.0056470
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056407, upper bound: 0.0056376
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056295, upper bound: 0.0056431
time: 0.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0056984, upper bound: 0.0056480
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0056908, upper bound: 0.0056535
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055773, upper bound: 0.0055908
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055783, upper bound: 0.0055675
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0054903, upper bound: 0.0054851
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0054854, upper bound: 0.0054851
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055268, upper bound: 0.0055200
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055265, upper bound: 0.0055200
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0054074, upper bound: 0.0054102
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0054074, upper bound: 0.0054136
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055685, upper bound: 0.0055832
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055633, upper bound: 0.0056188
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055792, upper bound: 0.0055799
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0055641, upper bound: 0.0055965
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0070916, upper bound: 0.0069085
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0069323, upper bound: 0.0070748
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0070916, upper bound: 0.0069043
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0069508, upper bound: 0.0071108
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0067563, upper bound: 0.0067563
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0067833, upper bound: 0.0071714
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0054020, upper bound: 0.0053998
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0053998, upper bound: 0.0053998
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0056434, upper bound: 0.0056448
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0056638, upper bound: 0.0056470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0056407, upper bound: 0.0056376
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0056295, upper bound: 0.0056431

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056858, upper bound: 0.0056293
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056429, upper bound: 0.0056293
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054822, upper bound: 0.0054805
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054805, upper bound: 0.0054805
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055522, upper bound: 0.0055516
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055516, upper bound: 0.0056032
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061345, upper bound: 0.0060108
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060828, upper bound: 0.0060108
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060540, upper bound: 0.0060108
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060540, upper bound: 0.0060144
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061522, upper bound: 0.0060108
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061016, upper bound: 0.0060108
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060544, upper bound: 0.0060108
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060544, upper bound: 0.0060196
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0058209, upper bound: 0.0058209
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0058209, upper bound: 0.0058209
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053963, upper bound: 0.0053963
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053963, upper bound: 0.0053963
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055516, upper bound: 0.0055530
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055516, upper bound: 0.0055522
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056605, upper bound: 0.0056367
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056293, upper bound: 0.0056293
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055081, upper bound: 0.0055097
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055081, upper bound: 0.0055081
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056293, upper bound: 0.0056293
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056293, upper bound: 0.0056429
time: 0.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0056858, upper bound: 0.0056293
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0056429, upper bound: 0.0056293
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0054822, upper bound: 0.0054805
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0054805, upper bound: 0.0054805
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0055522, upper bound: 0.0055516
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0055516, upper bound: 0.0056032
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0061345, upper bound: 0.0060108
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0060828, upper bound: 0.0060108
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0060540, upper bound: 0.0060108
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0060540, upper bound: 0.0060144
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0061522, upper bound: 0.0060108
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0061016, upper bound: 0.0060108
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0060544, upper bound: 0.0060108
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0060544, upper bound: 0.0060196
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0058209, upper bound: 0.0058209
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0058209, upper bound: 0.0058209
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0053963, upper bound: 0.0053963
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0053963, upper bound: 0.0053963
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0055516, upper bound: 0.0055530
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0055516, upper bound: 0.0055522
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0056605, upper bound: 0.0056367
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0056293, upper bound: 0.0056293
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0055081, upper bound: 0.0055097
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0055081, upper bound: 0.0055081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0056293, upper bound: 0.0056293
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.33
Output dim: 0, lower bound: -0.0056293, upper bound: 0.0056429

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054786, upper bound: 0.0054698
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055162, upper bound: 0.0055074
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055098, upper bound: 0.0055076
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055076
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055096, upper bound: 0.0055074
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055095, upper bound: 0.0055074
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055090
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055209
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055162, upper bound: 0.0055074
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055098, upper bound: 0.0055075
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055096, upper bound: 0.0055074
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055095, upper bound: 0.0055074
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055092
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055213
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053856, upper bound: 0.0053856
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0053856, upper bound: 0.0053856
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055375, upper bound: 0.0055375
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055375, upper bound: 0.0055375
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055375, upper bound: 0.0055422
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055709, upper bound: 0.0055447
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647
1: -0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365
2: -0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179
3: -0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761
4: -0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
time: 0.16 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054786, upper bound: 0.0054698
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055162, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055098, upper bound: 0.0055076
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055076
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055096, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055095, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055090
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055209
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055162, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055098, upper bound: 0.0055075
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055096, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055095, upper bound: 0.0055074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055092
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055092, upper bound: 0.0055213
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0053856, upper bound: 0.0053856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0053856, upper bound: 0.0053856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055375, upper bound: 0.0055375
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055375, upper bound: 0.0055375
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055375, upper bound: 0.0055422
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055709, upper bound: 0.0055447
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0055074, upper bound: 0.0055074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0054698, upper bound: 0.0054698

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.56 + 77.76 = 79.32 seconds
