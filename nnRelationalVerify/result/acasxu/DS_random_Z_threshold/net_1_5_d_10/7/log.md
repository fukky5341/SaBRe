## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 339.77104719722996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423)
1: (-124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621)
2: (-105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148)
3: (-110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960)
4: (-94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.07 + 2.28 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -339.8050277, upper bound: 339.8050277

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8047531, upper bound: 339.8050277
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8047531, upper bound: 339.8047531
time: 0.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 0, lower bound: -339.8047531, upper bound: 339.8050277
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 0, lower bound: -339.8047531, upper bound: 339.8047531

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7927413, upper bound: 339.7927448
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7927413, upper bound: 339.7927448
time: 0.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040480, upper bound: 339.8042984
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8044607, upper bound: 339.8040480
time: 0.80 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -339.7927413, upper bound: 339.7927448
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -339.7927413, upper bound: 339.7927448
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -339.8040480, upper bound: 339.8042984
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -339.8044607, upper bound: 339.8040480

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7927244, upper bound: 339.7927448
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7927413, upper bound: 339.7926265
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923016, upper bound: 339.7926027
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7926016, upper bound: 339.7925824
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8036176, upper bound: 339.8038859
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8036176, upper bound: 339.8037378
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8029433, upper bound: 339.8029433
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8036635, upper bound: 339.8029433
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.13 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.7927244, upper bound: 339.7927448
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.7927413, upper bound: 339.7926265
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.7923016, upper bound: 339.7926027
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.7926016, upper bound: 339.7925824
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.8036176, upper bound: 339.8038859
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.8036176, upper bound: 339.8037378
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.8029433, upper bound: 339.8029433
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -339.8036635, upper bound: 339.8029433

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922154, upper bound: 339.7924619
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923416, upper bound: 339.7922154
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920652, upper bound: 339.7920652
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920652, upper bound: 339.7920652
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923016, upper bound: 339.7926027
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923016, upper bound: 339.7923016
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7925632, upper bound: 339.7925192
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7926016, upper bound: 339.7923273
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8021007, upper bound: 339.8023610
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8021007, upper bound: 339.8023610
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8021007, upper bound: 339.8021497
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8021863, upper bound: 339.8021529
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8025557, upper bound: 339.8025470
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8025557, upper bound: 339.8025470
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023285, upper bound: 339.8016081
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016091, upper bound: 339.8016076
time: 0.87 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.76 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7922154, upper bound: 339.7924619
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7923416, upper bound: 339.7922154
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7920652, upper bound: 339.7920652
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7920652, upper bound: 339.7920652
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7923016, upper bound: 339.7926027
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7923016, upper bound: 339.7923016
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7925632, upper bound: 339.7925192
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.7926016, upper bound: 339.7923273
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8021007, upper bound: 339.8023610
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8021007, upper bound: 339.8023610
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8021007, upper bound: 339.8021497
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8021863, upper bound: 339.8021529
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8025557, upper bound: 339.8025470
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8025557, upper bound: 339.8025470
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8023285, upper bound: 339.8016081
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -339.8016091, upper bound: 339.8016076

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922006, upper bound: 339.7924333
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922024, upper bound: 339.7921871
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903192, upper bound: 339.7903064
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7905626
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7904127, upper bound: 339.7903891
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922594
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7925341, upper bound: 339.7922315
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7869852, upper bound: 339.7869689
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7869852, upper bound: 339.7869689
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8021832
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019231
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019714
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8020356, upper bound: 339.8019219
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016207, upper bound: 339.8016142
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016161, upper bound: 339.8016142
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023285, upper bound: 339.8016081
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8022276, upper bound: 339.8016076
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8008427, upper bound: 339.8008427
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8008427, upper bound: 339.8008427
time: 0.97 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7922006, upper bound: 339.7924333
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7922024, upper bound: 339.7921871
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7903192, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7905626
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7904127, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922594
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7925341, upper bound: 339.7922315
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7869852, upper bound: 339.7869689
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7869852, upper bound: 339.7869689
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8021832
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019231
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019714
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8020356, upper bound: 339.8019219
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8016207, upper bound: 339.8016142
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8016161, upper bound: 339.8016142
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8023285, upper bound: 339.8016081
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8022276, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8008427, upper bound: 339.8008427
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -339.8008427, upper bound: 339.8008427

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7924333
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7921871
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7921871
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922022, upper bound: 339.7921871
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903192, upper bound: 339.7903064
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7905626
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7904468
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922594
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7925341, upper bound: 339.7922315
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7857214, upper bound: 339.7857208
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8021832
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870406, upper bound: 339.7869274
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019714
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8014665, upper bound: 339.8015886
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8014487, upper bound: 339.8015886
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016142, upper bound: 339.8016142
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016142, upper bound: 339.8016142
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8022035, upper bound: 339.8016076
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023285, upper bound: 339.8016081
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016250, upper bound: 339.8016076
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.82 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7924333
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7921871
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7921871
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7922022, upper bound: 339.7921871
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7903192, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7905626
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7904468
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922594
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7925341, upper bound: 339.7922315
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7857214, upper bound: 339.7857208
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8021832
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7870406, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7870808, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019714
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8014665, upper bound: 339.8015886
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8014487, upper bound: 339.8015886
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8016142, upper bound: 339.8016142
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8016142, upper bound: 339.8016142
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8025470, upper bound: 339.8025470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8022035, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8023285, upper bound: 339.8016081
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8016250, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.02
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922006, upper bound: 339.7924333
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7922732
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835086, upper bound: 339.7835086
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835086, upper bound: 339.7835086
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7897812, upper bound: 339.7897812
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7897812, upper bound: 339.7897812
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7866455, upper bound: 339.7866455
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7866455, upper bound: 339.7866455
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7853833, upper bound: 339.7853833
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7853833, upper bound: 339.7853833
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7853491, upper bound: 339.7853491
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7853491, upper bound: 339.7853491
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727606, upper bound: 339.7727606
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727606, upper bound: 339.7727606
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837942, upper bound: 339.7837942
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837942, upper bound: 339.7837942
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7904468
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903844, upper bound: 339.7903844
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7904138, upper bound: 339.7903844
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922594
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918562, upper bound: 339.7918562
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918562, upper bound: 339.7918562
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7925341, upper bound: 339.7922315
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851748
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851625
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7855261, upper bound: 339.7855786
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7855365, upper bound: 339.7855261
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016838, upper bound: 339.8019552
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016838, upper bound: 339.8016972
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7920396
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7920399
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7842817, upper bound: 339.7842817
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7842817, upper bound: 339.7842817
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870351, upper bound: 339.7868998
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7868998, upper bound: 339.7868998
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870707, upper bound: 339.7868998
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7868998, upper bound: 339.7868998
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8014489, upper bound: 339.8016635
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8014487, upper bound: 339.8016920
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8009034, upper bound: 339.8009034
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8009034, upper bound: 339.8009034
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919998, upper bound: 339.7919998
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919998, upper bound: 339.7919998
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7992271, upper bound: 339.7992271
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7992271, upper bound: 339.7992271
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7989526, upper bound: 339.7989526
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7989530, upper bound: 339.7989526
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8008559, upper bound: 339.8008559
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8008559, upper bound: 339.8008559
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8012206, upper bound: 339.8012206
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8012206, upper bound: 339.8012206
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017394, upper bound: 339.8017394
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017394, upper bound: 339.8017394
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8012072, upper bound: 339.8012072
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8012072, upper bound: 339.8012072
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937165, upper bound: 339.7937165
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937165, upper bound: 339.7937165
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011613, upper bound: 339.8011613
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011613, upper bound: 339.8011613
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7951757, upper bound: 339.7951757
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7951757, upper bound: 339.7951757
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7964871, upper bound: 339.7964871
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7964871, upper bound: 339.7964871
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
time: 0.90 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.03 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7922006, upper bound: 339.7924333
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7922732
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7835086, upper bound: 339.7835086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7835086, upper bound: 339.7835086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7897812, upper bound: 339.7897812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7897812, upper bound: 339.7897812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7866455, upper bound: 339.7866455
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7866455, upper bound: 339.7866455
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7853833, upper bound: 339.7853833
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7853833, upper bound: 339.7853833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7853491, upper bound: 339.7853491
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7853491, upper bound: 339.7853491
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7727606, upper bound: 339.7727606
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7727606, upper bound: 339.7727606
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7837942, upper bound: 339.7837942
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7837942, upper bound: 339.7837942
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7904468
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7903844, upper bound: 339.7903844
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7904138, upper bound: 339.7903844
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922594
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7918562, upper bound: 339.7918562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7918562, upper bound: 339.7918562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7925341, upper bound: 339.7922315
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851748
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7855261, upper bound: 339.7855786
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7855365, upper bound: 339.7855261
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8016838, upper bound: 339.8019552
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8016838, upper bound: 339.8016972
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7920396
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7920399
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7842817, upper bound: 339.7842817
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7842817, upper bound: 339.7842817
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7870351, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7868998, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7870707, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7868998, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8014489, upper bound: 339.8016635
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8014487, upper bound: 339.8016920
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8009034, upper bound: 339.8009034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8009034, upper bound: 339.8009034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919998, upper bound: 339.7919998
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919998, upper bound: 339.7919998
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7992271, upper bound: 339.7992271
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7992271, upper bound: 339.7992271
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7989526, upper bound: 339.7989526
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7989530, upper bound: 339.7989526
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8008559, upper bound: 339.8008559
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8008559, upper bound: 339.8008559
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8012206, upper bound: 339.8012206
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8012206, upper bound: 339.8012206
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8017394, upper bound: 339.8017394
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8017394, upper bound: 339.8017394
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8012072, upper bound: 339.8012072
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8012072, upper bound: 339.8012072
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7937165, upper bound: 339.7937165
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7937165, upper bound: 339.7937165
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8011613, upper bound: 339.8011613
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.8011613, upper bound: 339.8011613
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7951757, upper bound: 339.7951757
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7951757, upper bound: 339.7951757
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7964871, upper bound: 339.7964871
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7964871, upper bound: 339.7964871
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851652, upper bound: 339.7851652
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851652, upper bound: 339.7851652
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7921871
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7922732
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7901362, upper bound: 339.7901362
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7901362, upper bound: 339.7901362
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
time: 1.04 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.63 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7851652, upper bound: 339.7851652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7851652, upper bound: 339.7851652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7921871
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7921871, upper bound: 339.7922732
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7918226, upper bound: 339.7918226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7901362, upper bound: 339.7901362
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7901362, upper bound: 339.7901362
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7918731, upper bound: 339.7918731
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7921473, upper bound: 339.7921473
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7835086, upper bound: 339.7835086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7835086, upper bound: 339.7835086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7903064, upper bound: 339.7903064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7897812, upper bound: 339.7897812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7897812, upper bound: 339.7897812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7899397, upper bound: 339.7899397
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7866455, upper bound: 339.7866455
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7866455, upper bound: 339.7866455
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7853833, upper bound: 339.7853833
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7853833, upper bound: 339.7853833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7853491, upper bound: 339.7853491
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7853491, upper bound: 339.7853491
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7884739, upper bound: 339.7884739
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7885136, upper bound: 339.7885136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7882388, upper bound: 339.7882388
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7916768, upper bound: 339.7916768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919246, upper bound: 339.7919246
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7727606, upper bound: 339.7727606
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7727606, upper bound: 339.7727606
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7753161, upper bound: 339.7753161
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7837942, upper bound: 339.7837942
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7837942, upper bound: 339.7837942
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7904468
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7903891, upper bound: 339.7903891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7903844, upper bound: 339.7903844
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7904138, upper bound: 339.7903844
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922594
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7918562, upper bound: 339.7918562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7918562, upper bound: 339.7918562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7922315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7925341, upper bound: 339.7922315
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851748
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851993
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7851625, upper bound: 339.7851625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7855261, upper bound: 339.7855786
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7855365, upper bound: 339.7855261
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7856838, upper bound: 339.7856838
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8016838, upper bound: 339.8019552
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8016838, upper bound: 339.8016972
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7920396
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7922437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7920396, upper bound: 339.7920399
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7869274, upper bound: 339.7869274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7842817, upper bound: 339.7842817
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7842817, upper bound: 339.7842817
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7870351, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7868998, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7870707, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7868998, upper bound: 339.7868998
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8019219, upper bound: 339.8019219
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8014489, upper bound: 339.8016635
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8014487, upper bound: 339.8016920
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8009034, upper bound: 339.8009034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8009034, upper bound: 339.8009034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919998, upper bound: 339.7919998
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919998, upper bound: 339.7919998
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7992271, upper bound: 339.7992271
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7992271, upper bound: 339.7992271
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7989526, upper bound: 339.7989526
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7989530, upper bound: 339.7989526
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8011666, upper bound: 339.8011666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8008559, upper bound: 339.8008559
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8008559, upper bound: 339.8008559
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8012206, upper bound: 339.8012206
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8012206, upper bound: 339.8012206
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8017394, upper bound: 339.8017394
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8017394, upper bound: 339.8017394
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7919502, upper bound: 339.7919502
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8016076, upper bound: 339.8016076
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8012072, upper bound: 339.8012072
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8012072, upper bound: 339.8012072
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7937165, upper bound: 339.7937165
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7937165, upper bound: 339.7937165
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8011613, upper bound: 339.8011613
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.8011613, upper bound: 339.8011613
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7951757, upper bound: 339.7951757
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7951757, upper bound: 339.7951757
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7964871, upper bound: 339.7964871
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7964871, upper bound: 339.7964871
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -339.7967033, upper bound: 339.7967033

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.35 + 417.35 = 420.70 seconds
