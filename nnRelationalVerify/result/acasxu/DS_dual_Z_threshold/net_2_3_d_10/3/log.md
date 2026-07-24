## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.12280842500000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436)
1: (0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292)
2: (0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002)
3: (-0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504)
4: (-0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 0.75 = 2.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1444805, upper bound: 0.1444805

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1356673, upper bound: 0.1356673
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1356673, upper bound: 0.1356673
time: 0.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.1356673, upper bound: 0.1356673
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.1356673, upper bound: 0.1356673

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1336612, upper bound: 0.1320588
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
time: 0.22 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1336612, upper bound: 0.1320588
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
time: 0.22 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -0.1336612, upper bound: 0.1320588
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -0.1336612, upper bound: 0.1320588
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1316573, upper bound: 0.1306871
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1307534, upper bound: 0.1306871
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1316573, upper bound: 0.1306871
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1307534, upper bound: 0.1306871
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
time: 0.23 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1316573, upper bound: 0.1306871
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1307534, upper bound: 0.1306871
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1316573, upper bound: 0.1306871
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1307534, upper bound: 0.1306871
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1312363, upper bound: 0.1300612
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1304350, upper bound: 0.1300391
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1304350
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1312363
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1312363, upper bound: 0.1300612
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1304350, upper bound: 0.1300391
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1304350
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1312363
time: 0.22 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1312363, upper bound: 0.1300612
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1304350, upper bound: 0.1300391
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1304350
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1312363
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1312363, upper bound: 0.1300612
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1304350, upper bound: 0.1300391
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1300612
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1300391
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300391, upper bound: 0.1304350
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.1300612, upper bound: 0.1312363

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1310954, upper bound: 0.1299072
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1301773, upper bound: 0.1299203
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1302941, upper bound: 0.1299000
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1301747, upper bound: 0.1299000
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299156, upper bound: 0.1299000
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299156
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1301747
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1302941
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1301773
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299072, upper bound: 0.1310954
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1310954, upper bound: 0.1299072
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1301773, upper bound: 0.1299203
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1302941, upper bound: 0.1299000
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1301747, upper bound: 0.1299000
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299156, upper bound: 0.1299000
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299156
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1301747
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1302941
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1301773
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1299072, upper bound: 0.1310954
time: 0.24 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1310954, upper bound: 0.1299072
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1301773, upper bound: 0.1299203
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1302941, upper bound: 0.1299000
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1301747, upper bound: 0.1299000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299156, upper bound: 0.1299000
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299156
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1301747
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1302941
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1301773
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299072, upper bound: 0.1310954
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1310954, upper bound: 0.1299072
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1301773, upper bound: 0.1299203
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1302941, upper bound: 0.1299000
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1301747, upper bound: 0.1299000
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299156, upper bound: 0.1299000
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299156
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299203
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1299000
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1299000
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1301747
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299000, upper bound: 0.1302941
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299203, upper bound: 0.1301773
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.1299072, upper bound: 0.1310954

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.93 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1275993, upper bound: 0.1263673
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1272290, upper bound: 0.1263970
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.93 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1266179, upper bound: 0.1264136
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263031, upper bound: 0.1264212
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.93 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1267714, upper bound: 0.1263160
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264834, upper bound: 0.1263661
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.93 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1266347, upper bound: 0.1264167
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263219, upper bound: 0.1264240
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.93 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264242, upper bound: 0.1263463
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264169, upper bound: 0.1263882
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.94 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264136
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263031, upper bound: 0.1264212
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.94 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264213, upper bound: 0.1263160
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264138, upper bound: 0.1263661
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.95 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264053, upper bound: 0.1264167
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263224, upper bound: 0.1264271
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.94 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264271, upper bound: 0.1263688
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1264077
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.95 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264138
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263031, upper bound: 0.1264213
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.95 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263161
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1263661
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.95 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263721, upper bound: 0.1264169
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263099, upper bound: 0.1264242
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.95 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264240, upper bound: 0.1263688
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1266608
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.96 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264963
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263129, upper bound: 0.1267793
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.95 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263161
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1266574
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.97 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263887, upper bound: 0.1272290
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263245, upper bound: 0.1275993
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.96 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1275993, upper bound: 0.1263245
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1272290, upper bound: 0.1263887
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.96 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1266574, upper bound: 0.1264136
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263161, upper bound: 0.1264212
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.96 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1267793, upper bound: 0.1263129
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264963, upper bound: 0.1263661
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.96 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1266608, upper bound: 0.1264167
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263688, upper bound: 0.1264240
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.97 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264242, upper bound: 0.1263099
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264169, upper bound: 0.1263721
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.96 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264136
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263160, upper bound: 0.1264212
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.97 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264213, upper bound: 0.1263031
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264138, upper bound: 0.1263661
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.97 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264077, upper bound: 0.1264167
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263688, upper bound: 0.1264271
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.97 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264271, upper bound: 0.1263224
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1264053
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.97 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264138
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263160, upper bound: 0.1264213
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.98 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263031
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1263661
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.98 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263882, upper bound: 0.1264169
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263463, upper bound: 0.1264242
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.98 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264240, upper bound: 0.1263219
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1266347
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.98 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264834
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263160, upper bound: 0.1267714
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.98 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263031
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1266179
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40
type: DSZ, layer: 5, pos: 42

Time for candidate selection: 0.98 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263970, upper bound: 0.1272290
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1263673, upper bound: 0.1275993
time: 0.24 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1275993, upper bound: 0.1263673
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1272290, upper bound: 0.1263970
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1266179, upper bound: 0.1264136
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263031, upper bound: 0.1264212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1267714, upper bound: 0.1263160
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264834, upper bound: 0.1263661
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1266347, upper bound: 0.1264167
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263219, upper bound: 0.1264240
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264242, upper bound: 0.1263463
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264169, upper bound: 0.1263882
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263031, upper bound: 0.1264212
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264213, upper bound: 0.1263160
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264138, upper bound: 0.1263661
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264053, upper bound: 0.1264167
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263224, upper bound: 0.1264271
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264271, upper bound: 0.1263688
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1264077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264138
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263031, upper bound: 0.1264213
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1263661
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263721, upper bound: 0.1264169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263099, upper bound: 0.1264242
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264240, upper bound: 0.1263688
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1266608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264963
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263129, upper bound: 0.1267793
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263161
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1266574
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263887, upper bound: 0.1272290
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263245, upper bound: 0.1275993
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1275993, upper bound: 0.1263245
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1272290, upper bound: 0.1263887
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1266574, upper bound: 0.1264136
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263161, upper bound: 0.1264212
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1267793, upper bound: 0.1263129
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264963, upper bound: 0.1263661
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1266608, upper bound: 0.1264167
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263688, upper bound: 0.1264240
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264242, upper bound: 0.1263099
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264169, upper bound: 0.1263721
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264136
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263160, upper bound: 0.1264212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264213, upper bound: 0.1263031
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264138, upper bound: 0.1263661
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264077, upper bound: 0.1264167
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263688, upper bound: 0.1264271
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264271, upper bound: 0.1263224
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1264053
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264138
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263160, upper bound: 0.1264213
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1263661
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263882, upper bound: 0.1264169
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263463, upper bound: 0.1264242
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264240, upper bound: 0.1263219
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264167, upper bound: 0.1266347
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263661, upper bound: 0.1264834
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263160, upper bound: 0.1267714
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264212, upper bound: 0.1263031
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1264136, upper bound: 0.1266179
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263970, upper bound: 0.1272290
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.1263673, upper bound: 0.1275993

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1270208, upper bound: 0.1252285
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1268124, upper bound: 0.1258902
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1266505, upper bound: 0.1253171
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1255981, upper bound: 0.1259202
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260704, upper bound: 0.1252391
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258571, upper bound: 0.1259439
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258334, upper bound: 0.1253088
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259516
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1262239, upper bound: 0.1250224
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260210, upper bound: 0.1258462
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260120, upper bound: 0.1252270
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251896, upper bound: 0.1258700
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260872, upper bound: 0.1252071
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258711, upper bound: 0.1259470
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258456, upper bound: 0.1252935
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1247394, upper bound: 0.1259543
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259544, upper bound: 0.1251486
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1253056, upper bound: 0.1258661
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259471, upper bound: 0.1252843
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251558, upper bound: 0.1259115
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258697, upper bound: 0.1252405
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252101, upper bound: 0.1259439
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258334, upper bound: 0.1253070
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259515
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259516, upper bound: 0.1249920
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1253328, upper bound: 0.1258459
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259440, upper bound: 0.1252225
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251528, upper bound: 0.1258708
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259099, upper bound: 0.1252143
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1254535, upper bound: 0.1259470
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258463, upper bound: 0.1252957
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1247551, upper bound: 0.1259570
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259570, upper bound: 0.1252524
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252285, upper bound: 0.1258875
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1254769
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251289, upper bound: 0.1259227
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258697, upper bound: 0.1252477
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251587, upper bound: 0.1259440
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258334, upper bound: 0.1253328
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259516
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1250259
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251966, upper bound: 0.1258460
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1252311
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1250651, upper bound: 0.1258700
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258777, upper bound: 0.1252009
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251587, upper bound: 0.1259471
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258381, upper bound: 0.1253056
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259544
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259543, upper bound: 0.1252771
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252714, upper bound: 0.1258868
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1259525
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251315, upper bound: 0.1261133
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258697, upper bound: 0.1254306
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251661, upper bound: 0.1260243
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258431, upper bound: 0.1260404
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1243484, upper bound: 0.1262319
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1250259
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252763, upper bound: 0.1258460
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1259511
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251116, upper bound: 0.1261099
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259009, upper bound: 0.1256786
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252775, upper bound: 0.1266505
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258548, upper bound: 0.1268152
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1247722, upper bound: 0.1270208
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1270208, upper bound: 0.1247722
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1268152, upper bound: 0.1258548
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1266505, upper bound: 0.1252775
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1256786, upper bound: 0.1259009
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1261099, upper bound: 0.1251116
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259511, upper bound: 0.1259439
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258460, upper bound: 0.1252763
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1250259, upper bound: 0.1259515
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1262319, upper bound: 0.1243484
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260404, upper bound: 0.1258431
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260243, upper bound: 0.1251661
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1254306, upper bound: 0.1258697
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1261133, upper bound: 0.1251315
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259525, upper bound: 0.1259470
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258868, upper bound: 0.1252714
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252771, upper bound: 0.1259543
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259544, upper bound: 0.1243384
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1253056, upper bound: 0.1258381
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259471, upper bound: 0.1251587
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252009, upper bound: 0.1258777
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258700, upper bound: 0.1250651
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252311, upper bound: 0.1259439
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258459, upper bound: 0.1251966
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1250259, upper bound: 0.1259515
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259516, upper bound: 0.1243384
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1253328, upper bound: 0.1258334
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259440, upper bound: 0.1251587
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252477, upper bound: 0.1258697
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259227, upper bound: 0.1251289
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1254769, upper bound: 0.1259470
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258875, upper bound: 0.1252285
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252524, upper bound: 0.1259570
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259570, upper bound: 0.1247551
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252957, upper bound: 0.1258463
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1254535
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252143, upper bound: 0.1259099
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258708, upper bound: 0.1251528
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252225, upper bound: 0.1259440
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258459, upper bound: 0.1253328
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1249920, upper bound: 0.1259516
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1243384
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1253070, upper bound: 0.1258334
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1252101
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252405, upper bound: 0.1258697
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259115, upper bound: 0.1251558
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252843, upper bound: 0.1259471
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258661, upper bound: 0.1253056
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251486, upper bound: 0.1259544
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259543, upper bound: 0.1247394
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252935, upper bound: 0.1258456
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1258711
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252071, upper bound: 0.1260872
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258700, upper bound: 0.1251896
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252270, upper bound: 0.1260120
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258462, upper bound: 0.1260210
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1250224, upper bound: 0.1262239
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1243384
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1253088, upper bound: 0.1258334
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1258571
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252391, upper bound: 0.1260704
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1259202, upper bound: 0.1255981
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1253171, upper bound: 0.1266505
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 15
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 15

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 33

### Candidate
type: DSZ, layer: 5, pos: 40

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 36
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 7, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258902, upper bound: 0.1268124
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252285, upper bound: 0.1270208
time: 0.25 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1270208, upper bound: 0.1252285
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1268124, upper bound: 0.1258902
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1266505, upper bound: 0.1253171
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1255981, upper bound: 0.1259202
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1260704, upper bound: 0.1252391
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258571, upper bound: 0.1259439
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258334, upper bound: 0.1253088
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259516
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1262239, upper bound: 0.1250224
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1260210, upper bound: 0.1258462
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1260120, upper bound: 0.1252270
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251896, upper bound: 0.1258700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1260872, upper bound: 0.1252071
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258711, upper bound: 0.1259470
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258456, upper bound: 0.1252935
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1247394, upper bound: 0.1259543
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259544, upper bound: 0.1251486
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1253056, upper bound: 0.1258661
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259471, upper bound: 0.1252843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251558, upper bound: 0.1259115
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258697, upper bound: 0.1252405
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252101, upper bound: 0.1259439
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258334, upper bound: 0.1253070
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259515
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259516, upper bound: 0.1249920
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1253328, upper bound: 0.1258459
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259440, upper bound: 0.1252225
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251528, upper bound: 0.1258708
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259099, upper bound: 0.1252143
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1254535, upper bound: 0.1259470
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258463, upper bound: 0.1252957
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1247551, upper bound: 0.1259570
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259570, upper bound: 0.1252524
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252285, upper bound: 0.1258875
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1254769
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251289, upper bound: 0.1259227
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258697, upper bound: 0.1252477
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251587, upper bound: 0.1259440
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258334, upper bound: 0.1253328
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259516
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1250259
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251966, upper bound: 0.1258460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1252311
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1250651, upper bound: 0.1258700
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258777, upper bound: 0.1252009
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251587, upper bound: 0.1259471
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258381, upper bound: 0.1253056
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1259544
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259543, upper bound: 0.1252771
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252714, upper bound: 0.1258868
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1259525
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251315, upper bound: 0.1261133
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258697, upper bound: 0.1254306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251661, upper bound: 0.1260243
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258431, upper bound: 0.1260404
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1243484, upper bound: 0.1262319
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1250259
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252763, upper bound: 0.1258460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1259511
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251116, upper bound: 0.1261099
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259009, upper bound: 0.1256786
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252775, upper bound: 0.1266505
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258548, upper bound: 0.1268152
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1247722, upper bound: 0.1270208
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1270208, upper bound: 0.1247722
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1268152, upper bound: 0.1258548
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1266505, upper bound: 0.1252775
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1256786, upper bound: 0.1259009
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1261099, upper bound: 0.1251116
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259511, upper bound: 0.1259439
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258460, upper bound: 0.1252763
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1250259, upper bound: 0.1259515
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1262319, upper bound: 0.1243484
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1260404, upper bound: 0.1258431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1260243, upper bound: 0.1251661
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1254306, upper bound: 0.1258697
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1261133, upper bound: 0.1251315
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259525, upper bound: 0.1259470
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258868, upper bound: 0.1252714
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252771, upper bound: 0.1259543
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259544, upper bound: 0.1243384
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1253056, upper bound: 0.1258381
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259471, upper bound: 0.1251587
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252009, upper bound: 0.1258777
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258700, upper bound: 0.1250651
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252311, upper bound: 0.1259439
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258459, upper bound: 0.1251966
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1250259, upper bound: 0.1259515
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259516, upper bound: 0.1243384
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1253328, upper bound: 0.1258334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259440, upper bound: 0.1251587
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252477, upper bound: 0.1258697
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259227, upper bound: 0.1251289
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1254769, upper bound: 0.1259470
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258875, upper bound: 0.1252285
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252524, upper bound: 0.1259570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259570, upper bound: 0.1247551
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252957, upper bound: 0.1258463
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1254535
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252143, upper bound: 0.1259099
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258708, upper bound: 0.1251528
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252225, upper bound: 0.1259440
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258459, upper bound: 0.1253328
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1249920, upper bound: 0.1259516
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1243384
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1253070, upper bound: 0.1258334
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1252101
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252405, upper bound: 0.1258697
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259115, upper bound: 0.1251558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252843, upper bound: 0.1259471
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258661, upper bound: 0.1253056
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1251486, upper bound: 0.1259544
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259543, upper bound: 0.1247394
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252935, upper bound: 0.1258456
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259470, upper bound: 0.1258711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252071, upper bound: 0.1260872
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258700, upper bound: 0.1251896
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252270, upper bound: 0.1260120
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258462, upper bound: 0.1260210
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1250224, upper bound: 0.1262239
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259515, upper bound: 0.1243384
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1253088, upper bound: 0.1258334
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259439, upper bound: 0.1258571
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252391, upper bound: 0.1260704
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1259202, upper bound: 0.1255981
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1253171, upper bound: 0.1266505
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1258902, upper bound: 0.1268124
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.1252285, upper bound: 0.1270208

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251404, upper bound: 0.1252285
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1270208, upper bound: 0.1235010
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1252304, upper bound: 0.1258902
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1268124, upper bound: 0.1235471
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1243392, upper bound: 0.1253171
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1266505, upper bound: 0.1235768
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1241503, upper bound: 0.1259202
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1255981, upper bound: 0.1235874
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1238835, upper bound: 0.1252391
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260704, upper bound: 0.1235188
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1241193, upper bound: 0.1259439
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258571, upper bound: 0.1235748
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1233333, upper bound: 0.1253088
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1258334, upper bound: 0.1235800
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 7, pos: 39

### Candidate
type: DSZ, layer: 7, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1230411, upper bound: 0.1259516
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1243384, upper bound: 0.1235915
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 7
type: DSZ, layer: 7, pos: 39
type: DSZ, layer: 7, pos: 25
type: DSZ, layer: 7, pos: 22
type: DSZ, layer: 7, pos: 27
type: DSZ, layer: 7, pos: 6
type: DSZ, layer: 7, pos: 47

Time for candidate selection: 0.14 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.28 + 417.84 = 420.13 seconds
