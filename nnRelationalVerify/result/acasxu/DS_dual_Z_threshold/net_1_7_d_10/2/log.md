## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 8.0073e-05


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361)
1: (-0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682)
2: (-0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192)
3: (-0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065)
4: (-0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.90 + 0.56 = 1.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.14 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.36 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.36
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.36
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.15 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.17 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.14 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.23
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000741
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361
1: -0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682
2: -0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192
3: -0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065
4: -0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
time: 0.15 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.31 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000741
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000742

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.46 + 77.86 = 79.32 seconds
