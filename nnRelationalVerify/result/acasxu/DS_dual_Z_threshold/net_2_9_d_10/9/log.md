## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 912.9840697103999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561)
1: (-47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646)
2: (-38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152)
3: (-44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547)
4: (-33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.02 + 1.53 = 4.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -916.6506724, upper bound: 916.6506724

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5808157, upper bound: 913.5808157
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5808157, upper bound: 913.5808157
time: 0.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 0, lower bound: -913.5808157, upper bound: 913.5808157
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 0, lower bound: -913.5808157, upper bound: 913.5808157

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685
time: 0.45 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685
time: 0.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -913.0668685, upper bound: 913.0668685

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
time: 0.45 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 0, lower bound: -913.0272096, upper bound: 913.0272096

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561
1: -47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646
2: -38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152
3: -44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547
4: -33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729

Time for backsubstitution: 2.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
time: 0.53 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.02
Output dim: 0, lower bound: -912.9396498, upper bound: 912.9396498

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.55 + 60.73 = 65.28 seconds
