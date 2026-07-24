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
execution time: IAR + RelationalAnalysis = 0.60 + 0.66 = 1.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1444805, upper bound: 0.1444805

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1433590, upper bound: 0.1370225
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1370225, upper bound: 0.1433590
time: 0.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.33 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.33
Output dim: 0, lower bound: -0.1433590, upper bound: 0.1370225
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.33
Output dim: 0, lower bound: -0.1370225, upper bound: 0.1433590

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1332273, upper bound: 0.1324916
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1332273, upper bound: 0.1324916
time: 0.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 0.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 0.91
Output dim: 0, lower bound: -0.1332273, upper bound: 0.1324916
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.91
Output dim: 0, lower bound: -0.1332273, upper bound: 0.1324916
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 0.91
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.91
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1275428, upper bound: 0.1260151
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.06
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1261680, upper bound: 0.1247976
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260289, upper bound: 0.1247976
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1248218, upper bound: 0.1242952
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251975, upper bound: 0.1242952
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1250938, upper bound: 0.1242952
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1251975, upper bound: 0.1242952
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1153386, upper bound: 0.1153386
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1153386, upper bound: 0.1153386
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1242952
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1250938
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1248218
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1250938
time: 0.14 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 0.92 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1261680, upper bound: 0.1247976
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1260289, upper bound: 0.1247976
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1248218, upper bound: 0.1242952
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1251975, upper bound: 0.1242952
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1250938, upper bound: 0.1242952
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1251975, upper bound: 0.1242952
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1153386, upper bound: 0.1153386
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1153386, upper bound: 0.1153386
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1242952
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1250938
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1248218
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1251975
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.92
Output dim: 0, lower bound: -0.1242952, upper bound: 0.1250938

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1239262, upper bound: 0.1230239
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1140751, upper bound: 0.1140751
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1140751, upper bound: 0.1140751
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1238225, upper bound: 0.1230239
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1238225
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1238225
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239
time: 0.17 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 0.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1239262, upper bound: 0.1230239
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1140751, upper bound: 0.1140751
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1140751, upper bound: 0.1140751
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1238225, upper bound: 0.1230239
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1238225
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1137731
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1238225
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.97
Output dim: 0, lower bound: -0.1230239, upper bound: 0.1230239

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 11
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 11

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436
1: 0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292
2: 0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002
3: -0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504
4: -0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
time: 0.17 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.1125097, upper bound: 0.1125097

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.26 + 38.11 = 39.37 seconds
