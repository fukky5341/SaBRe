## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 607.026092835328


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574)
1: (-264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331)
2: (-213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147)
3: (-300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574)
4: (-274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.12 + 2.36 = 3.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -607.0382336, upper bound: 607.0382336

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0345642, upper bound: 607.0345642
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0345642, upper bound: 607.0345642
time: 0.92 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -607.0345642, upper bound: 607.0345642
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -607.0345642, upper bound: 607.0345642

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0274133, upper bound: 607.0274140
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0274133, upper bound: 607.0274137
time: 0.76 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0345618, upper bound: 607.0345642
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0345642, upper bound: 607.0345618
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.77 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -607.0274133, upper bound: 607.0274140
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -607.0274133, upper bound: 607.0274137
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -607.0345618, upper bound: 607.0345642
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -607.0345642, upper bound: 607.0345618

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0274133, upper bound: 607.0274131
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273976, upper bound: 607.0274140
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0255224, upper bound: 607.0255450
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0255224, upper bound: 607.0255450
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0303096, upper bound: 607.0303095
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0303096, upper bound: 607.0303095
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0345640, upper bound: 607.0345240
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0345632, upper bound: 607.0345615
time: 1.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.23 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0274133, upper bound: 607.0274131
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0273976, upper bound: 607.0274140
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0255224, upper bound: 607.0255450
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0255224, upper bound: 607.0255450
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0303096, upper bound: 607.0303095
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0303096, upper bound: 607.0303095
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0345640, upper bound: 607.0345240
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -607.0345632, upper bound: 607.0345615

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273546, upper bound: 607.0274113
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273819, upper bound: 607.0274133
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267878, upper bound: 607.0267989
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267878, upper bound: 607.0267982
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302675, upper bound: 607.0302900
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302673, upper bound: 607.0302671
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0262586, upper bound: 607.0262182
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0262586, upper bound: 607.0261933
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0344503, upper bound: 607.0344100
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0344421, upper bound: 607.0344100
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0323066, upper bound: 607.0323082
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0323066, upper bound: 607.0323082
time: 1.01 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0273546, upper bound: 607.0274113
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0273819, upper bound: 607.0274133
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0267878, upper bound: 607.0267989
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0267878, upper bound: 607.0267982
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0302675, upper bound: 607.0302900
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0302673, upper bound: 607.0302671
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0262586, upper bound: 607.0262182
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0262586, upper bound: 607.0261933
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0344503, upper bound: 607.0344100
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0344421, upper bound: 607.0344100
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0323066, upper bound: 607.0323082
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -607.0323066, upper bound: 607.0323082

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267820, upper bound: 607.0268042
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267820, upper bound: 607.0268288
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273546, upper bound: 607.0274115
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273546, upper bound: 607.0274133
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263877, upper bound: 607.0264086
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0246625, upper bound: 607.0247055
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0246625, upper bound: 607.0247055
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301684, upper bound: 607.0301684
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301684, upper bound: 607.0301935
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260125, upper bound: 607.0259402
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0259316, upper bound: 607.0259884
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0261577, upper bound: 607.0261954
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0262337, upper bound: 607.0261549
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0249658, upper bound: 607.0249658
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0249814, upper bound: 607.0249658
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0328148, upper bound: 607.0327968
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0328392, upper bound: 607.0327968
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272816, upper bound: 607.0272666
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272842, upper bound: 607.0272666
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322263, upper bound: 607.0322168
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322153, upper bound: 607.0322153
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0318839, upper bound: 607.0318738
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0318764, upper bound: 607.0318760
time: 0.79 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.79 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0267820, upper bound: 607.0268042
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0267820, upper bound: 607.0268288
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0273546, upper bound: 607.0274115
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0273546, upper bound: 607.0274133
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0263877, upper bound: 607.0264086
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0246625, upper bound: 607.0247055
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0246625, upper bound: 607.0247055
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0301684, upper bound: 607.0301684
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0301684, upper bound: 607.0301935
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0260125, upper bound: 607.0259402
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0259316, upper bound: 607.0259884
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0261577, upper bound: 607.0261954
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0262337, upper bound: 607.0261549
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0249658, upper bound: 607.0249658
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0249814, upper bound: 607.0249658
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0328148, upper bound: 607.0327968
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0328392, upper bound: 607.0327968
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0272816, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0272842, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0322263, upper bound: 607.0322168
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0322153, upper bound: 607.0322153
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0318839, upper bound: 607.0318738
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -607.0318764, upper bound: 607.0318760

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0236543, upper bound: 607.0236543
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0236543, upper bound: 607.0236543
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267058, upper bound: 607.0267502
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267058, upper bound: 607.0267520
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272868, upper bound: 607.0273156
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273147, upper bound: 607.0273447
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272569, upper bound: 607.0272827
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272569, upper bound: 607.0272569
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263853
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263563, upper bound: 607.0263480
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301704, upper bound: 607.0301666
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301706, upper bound: 607.0301684
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0266998, upper bound: 607.0267187
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0266970, upper bound: 607.0266970
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0261019, upper bound: 607.0260813
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260422, upper bound: 607.0260895
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0249718, upper bound: 607.0249391
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0249352, upper bound: 607.0249359
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310825, upper bound: 607.0310825
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310825, upper bound: 607.0310825
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273115, upper bound: 607.0272666
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272842, upper bound: 607.0272666
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271501, upper bound: 607.0271469
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272842, upper bound: 607.0272666
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322323, upper bound: 607.0322168
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322153, upper bound: 607.0322153
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280940, upper bound: 607.0280940
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280946, upper bound: 607.0280940
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298448, upper bound: 607.0298448
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298459, upper bound: 607.0298448
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0318716, upper bound: 607.0318670
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0318648, upper bound: 607.0318648
time: 0.81 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.75 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0236543, upper bound: 607.0236543
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0236543, upper bound: 607.0236543
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0267058, upper bound: 607.0267502
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0267058, upper bound: 607.0267520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0272868, upper bound: 607.0273156
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0273147, upper bound: 607.0273447
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0272569, upper bound: 607.0272827
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0272569, upper bound: 607.0272569
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263853
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0263563, upper bound: 607.0263480
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0301704, upper bound: 607.0301666
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0301706, upper bound: 607.0301684
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0266998, upper bound: 607.0267187
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0266970, upper bound: 607.0266970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0261019, upper bound: 607.0260813
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0260422, upper bound: 607.0260895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0249718, upper bound: 607.0249391
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0249352, upper bound: 607.0249359
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0310825, upper bound: 607.0310825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0310825, upper bound: 607.0310825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0273115, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0272842, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0271501, upper bound: 607.0271469
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0272842, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0322323, upper bound: 607.0322168
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0322153, upper bound: 607.0322153
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0280940, upper bound: 607.0280940
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0280946, upper bound: 607.0280940
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0298448, upper bound: 607.0298448
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0298459, upper bound: 607.0298448
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0318716, upper bound: 607.0318670
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -607.0318648, upper bound: 607.0318648

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264426
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264946
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267057, upper bound: 607.0267335
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267057, upper bound: 607.0267520
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0271962
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0272242
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0228599, upper bound: 607.0228599
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0228599, upper bound: 607.0229094
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270737, upper bound: 607.0270812
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270702, upper bound: 607.0270897
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272568, upper bound: 607.0272568
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272568, upper bound: 607.0272568
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263853
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263850
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263460
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263480
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0217629, upper bound: 607.0217629
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0217629, upper bound: 607.0218260
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301568, upper bound: 607.0301568
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301585, upper bound: 607.0301568
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299900, upper bound: 607.0299898
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299899, upper bound: 607.0299895
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0264839
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0265120
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0244932, upper bound: 607.0244932
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0244932, upper bound: 607.0244932
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260246, upper bound: 607.0260611
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260246, upper bound: 607.0260246
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280459, upper bound: 607.0280459
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280459, upper bound: 607.0280459
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310825, upper bound: 607.0310825
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310989, upper bound: 607.0310825
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267098, upper bound: 607.0267075
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271684, upper bound: 607.0271374
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271374, upper bound: 607.0271374
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0228119, upper bound: 607.0228119
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0228119, upper bound: 607.0228278
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0249013, upper bound: 607.0249079
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0249013, upper bound: 607.0249079
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317839, upper bound: 607.0317839
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317839, upper bound: 607.0317839
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280940, upper bound: 607.0280940
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280940, upper bound: 607.0280940
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298448, upper bound: 607.0298448
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298448, upper bound: 607.0298448
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298292, upper bound: 607.0298292
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298292, upper bound: 607.0298292
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317825, upper bound: 607.0317770
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317824, upper bound: 607.0317751
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0254172, upper bound: 607.0254236
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0254172, upper bound: 607.0254236
time: 1.06 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264426
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264946
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0267057, upper bound: 607.0267335
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0267057, upper bound: 607.0267520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0271962
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0272242
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0228599, upper bound: 607.0228599
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0228599, upper bound: 607.0229094
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0270737, upper bound: 607.0270812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0270702, upper bound: 607.0270897
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0272568, upper bound: 607.0272568
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0272568, upper bound: 607.0272568
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263853
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263850
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263480
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0217629, upper bound: 607.0217629
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0217629, upper bound: 607.0218260
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0301568, upper bound: 607.0301568
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0301585, upper bound: 607.0301568
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0299900, upper bound: 607.0299898
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0299899, upper bound: 607.0299895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0264839
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0265120
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0244932, upper bound: 607.0244932
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0244932, upper bound: 607.0244932
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0260246, upper bound: 607.0260611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0260246, upper bound: 607.0260246
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0280459, upper bound: 607.0280459
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0280459, upper bound: 607.0280459
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0310825, upper bound: 607.0310825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0310989, upper bound: 607.0310825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0267098, upper bound: 607.0267075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0271684, upper bound: 607.0271374
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0271374, upper bound: 607.0271374
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0228119, upper bound: 607.0228119
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0228119, upper bound: 607.0228278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0249013, upper bound: 607.0249079
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0249013, upper bound: 607.0249079
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0317839, upper bound: 607.0317839
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0317839, upper bound: 607.0317839
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0280940, upper bound: 607.0280940
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0280940, upper bound: 607.0280940
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0298448, upper bound: 607.0298448
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0298448, upper bound: 607.0298448
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0298292, upper bound: 607.0298292
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0298292, upper bound: 607.0298292
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0317825, upper bound: 607.0317770
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0317824, upper bound: 607.0317751
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0254172, upper bound: 607.0254236
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 0, lower bound: -607.0254172, upper bound: 607.0254236

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260924, upper bound: 607.0260924
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260923, upper bound: 607.0260923
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264484
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264946
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265804, upper bound: 607.0265881
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265804, upper bound: 607.0266146
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0228151, upper bound: 607.0228150
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0228151, upper bound: 607.0228837
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0227257, upper bound: 607.0227257
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0227257, upper bound: 607.0227269
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0272242
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0271962
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270702, upper bound: 607.0270797
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270737, upper bound: 607.0270812
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0230680, upper bound: 607.0230680
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0230680, upper bound: 607.0230680
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270701, upper bound: 607.0270701
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270701, upper bound: 607.0270701
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272527, upper bound: 607.0272527
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272564, upper bound: 607.0272527
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263790
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263626
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263250, upper bound: 607.0263749
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263250, upper bound: 607.0263722
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263452
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263460
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263445
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263349
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0263677
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0244700, upper bound: 607.0245151
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0244700, upper bound: 607.0245151
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0296173, upper bound: 607.0296173
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0296173, upper bound: 607.0296173
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301005, upper bound: 607.0300995
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300995, upper bound: 607.0300995
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0293480, upper bound: 607.0293183
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0293183, upper bound: 607.0293398
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0297479, upper bound: 607.0297479
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0297479, upper bound: 607.0297479
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264468, upper bound: 607.0264750
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264788, upper bound: 607.0264693
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0264522
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0265120
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0247898, upper bound: 607.0247898
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0247898, upper bound: 607.0247898
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0277130, upper bound: 607.0277130
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0277130, upper bound: 607.0277130
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0306786, upper bound: 607.0306786
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0306786, upper bound: 607.0306786
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310613, upper bound: 607.0310613
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310861, upper bound: 607.0310613
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267134, upper bound: 607.0267075
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0218488, upper bound: 607.0218488
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0218488, upper bound: 607.0218488
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0260956, upper bound: 607.0260956
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0261799, upper bound: 607.0260956
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0266556, upper bound: 607.0266556
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0266865, upper bound: 607.0266556
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267004, upper bound: 607.0267004
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267004, upper bound: 607.0267004
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0237177, upper bound: 607.0237177
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0237177, upper bound: 607.0237177
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271469
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271469
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271469
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272646, upper bound: 607.0272646
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272646, upper bound: 607.0272646
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0251922, upper bound: 607.0251922
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0251922, upper bound: 607.0251922
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0279198, upper bound: 607.0279198
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0279198, upper bound: 607.0279198
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280651, upper bound: 607.0280651
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280651, upper bound: 607.0280651
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280669, upper bound: 607.0280669
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280669, upper bound: 607.0280669
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280827, upper bound: 607.0280827
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280827, upper bound: 607.0280827
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0296779, upper bound: 607.0296779
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0296779, upper bound: 607.0296785
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0248480, upper bound: 607.0248457
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0248480, upper bound: 607.0248457
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0292281, upper bound: 607.0292281
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0292281, upper bound: 607.0292281
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0279980, upper bound: 607.0279919
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0279999, upper bound: 607.0279919
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317497, upper bound: 607.0317426
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317492, upper bound: 607.0317447
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0311183, upper bound: 607.0311183
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0311183, upper bound: 607.0311183
time: 0.83 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0260924, upper bound: 607.0260924
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0260923, upper bound: 607.0260923
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264484
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264946
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0265804, upper bound: 607.0265881
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0265804, upper bound: 607.0266146
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0228151, upper bound: 607.0228150
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0228151, upper bound: 607.0228837
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0227257, upper bound: 607.0227257
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0227257, upper bound: 607.0227269
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0272242
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0271962
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0270702, upper bound: 607.0270797
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0270737, upper bound: 607.0270812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0230680, upper bound: 607.0230680
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0230680, upper bound: 607.0230680
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0270701, upper bound: 607.0270701
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0270701, upper bound: 607.0270701
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0272527, upper bound: 607.0272527
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0272564, upper bound: 607.0272527
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263790
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263626
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263250, upper bound: 607.0263749
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263250, upper bound: 607.0263722
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263452
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263362, upper bound: 607.0263460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263445
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263342, upper bound: 607.0263349
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0264115
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0263677, upper bound: 607.0263677
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0244700, upper bound: 607.0245151
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0244700, upper bound: 607.0245151
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0296173, upper bound: 607.0296173
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0296173, upper bound: 607.0296173
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0301005, upper bound: 607.0300995
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0300995, upper bound: 607.0300995
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0293480, upper bound: 607.0293183
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0293183, upper bound: 607.0293398
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0297479, upper bound: 607.0297479
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0297479, upper bound: 607.0297479
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0264468, upper bound: 607.0264750
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0264788, upper bound: 607.0264693
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0264522
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0264507, upper bound: 607.0265120
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0247898, upper bound: 607.0247898
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0247898, upper bound: 607.0247898
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0277130, upper bound: 607.0277130
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0277130, upper bound: 607.0277130
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0306786, upper bound: 607.0306786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0306786, upper bound: 607.0306786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0310613, upper bound: 607.0310613
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0310861, upper bound: 607.0310613
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0267075, upper bound: 607.0267075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0267134, upper bound: 607.0267075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0218488, upper bound: 607.0218488
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0218488, upper bound: 607.0218488
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0260956, upper bound: 607.0260956
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0261799, upper bound: 607.0260956
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0266556, upper bound: 607.0266556
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0266865, upper bound: 607.0266556
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0267004, upper bound: 607.0267004
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0267004, upper bound: 607.0267004
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0237177, upper bound: 607.0237177
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0237177, upper bound: 607.0237177
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271469
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271469
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271470
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0271470, upper bound: 607.0271469
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0272666, upper bound: 607.0272666
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0272646, upper bound: 607.0272646
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0272646, upper bound: 607.0272646
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0251922, upper bound: 607.0251922
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0251922, upper bound: 607.0251922
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0279198, upper bound: 607.0279198
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0279198, upper bound: 607.0279198
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280651, upper bound: 607.0280651
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280651, upper bound: 607.0280651
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280926, upper bound: 607.0280926
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280669, upper bound: 607.0280669
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280669, upper bound: 607.0280669
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280827, upper bound: 607.0280827
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0280827, upper bound: 607.0280827
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0296779, upper bound: 607.0296779
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0296779, upper bound: 607.0296785
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0248480, upper bound: 607.0248457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0248480, upper bound: 607.0248457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0292281, upper bound: 607.0292281
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0292281, upper bound: 607.0292281
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0279980, upper bound: 607.0279919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0279999, upper bound: 607.0279919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0317497, upper bound: 607.0317426
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0317492, upper bound: 607.0317447
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0311183, upper bound: 607.0311183
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -607.0311183, upper bound: 607.0311183

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264482
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264484
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264946
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264426, upper bound: 607.0264905
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265804, upper bound: 607.0265864
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265804, upper bound: 607.0265881
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265740, upper bound: 607.0266143
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265740, upper bound: 607.0265739
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271947, upper bound: 607.0272220
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271947, upper bound: 607.0272008
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0271962
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0271962, upper bound: 607.0271962
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270701, upper bound: 607.0270788
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270701, upper bound: 607.0270701
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270405, upper bound: 607.0270547
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270450, upper bound: 607.0270450
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270701, upper bound: 607.0270701
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270782, upper bound: 607.0270701
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270400, upper bound: 607.0270400
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270400, upper bound: 607.0270400
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267706, upper bound: 607.0267706
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267706, upper bound: 607.0267706
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272527, upper bound: 607.0272527
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0272527, upper bound: 607.0272527
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263249, upper bound: 607.0263704
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263249, upper bound: 607.0263661
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0244355, upper bound: 607.0244439
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0244355, upper bound: 607.0244355
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260495, upper bound: 607.0260495
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0260495, upper bound: 607.0260866
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263250, upper bound: 607.0263250
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263250, upper bound: 607.0263722
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0261986, upper bound: 607.0262007
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0261986, upper bound: 607.0261986
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574
1: -264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331
2: -213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147
3: -300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574
4: -274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110

Time for backsubstitution: 1.23 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.48 + 416.85 = 420.33 seconds
