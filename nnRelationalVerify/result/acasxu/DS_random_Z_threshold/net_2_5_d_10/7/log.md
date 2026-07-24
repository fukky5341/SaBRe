## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 655.396249165482


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729)
1: (-107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362)
2: (-66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734)
3: (-134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476)
4: (-73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 1.94 = 2.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -655.4355753, upper bound: 655.4355753

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4345000
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4343687
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4345000
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4343687

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343612, upper bound: 655.4344781
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343612, upper bound: 655.4345000
time: 0.67 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343589, upper bound: 655.4342680
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344115, upper bound: 655.4342880
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -655.4343612, upper bound: 655.4344781
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -655.4343612, upper bound: 655.4345000
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -655.4343589, upper bound: 655.4342680
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -655.4344115, upper bound: 655.4342880

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4340012
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4339983
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342563, upper bound: 655.4343952
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342567, upper bound: 655.4343148
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323328, upper bound: 655.4322755
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323328, upper bound: 655.4322755
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4312237, upper bound: 655.4310546
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4310546, upper bound: 655.4310546
time: 0.63 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.00 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4340012
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4339983
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4342563, upper bound: 655.4343952
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4342567, upper bound: 655.4343148
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4323328, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4323328, upper bound: 655.4322755
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4312237, upper bound: 655.4310546
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -655.4310546, upper bound: 655.4310546

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334238, upper bound: 655.4334238
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334238, upper bound: 655.4334238
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4339983
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4339325
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342553, upper bound: 655.4343952
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342563, upper bound: 655.4342553
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341260, upper bound: 655.4341950
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341260, upper bound: 655.4341260
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323262, upper bound: 655.4322755
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323328, upper bound: 655.4322755
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4310546, upper bound: 655.4310546
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4310546, upper bound: 655.4310546
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4310085, upper bound: 655.4309467
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4334238, upper bound: 655.4334238
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4334238, upper bound: 655.4334238
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4339983
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4338915, upper bound: 655.4339325
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4342553, upper bound: 655.4343952
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4342563, upper bound: 655.4342553
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4341260, upper bound: 655.4341950
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4341260, upper bound: 655.4341260
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4323262, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4323328, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4310546, upper bound: 655.4310546
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4310546, upper bound: 655.4310546
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -655.4310085, upper bound: 655.4309467

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333839, upper bound: 655.4333839
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333839, upper bound: 655.4333839
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4310209
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338577, upper bound: 655.4339844
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338577, upper bound: 655.4338577
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337458, upper bound: 655.4337786
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337458, upper bound: 655.4337458
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341260, upper bound: 655.4342747
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341259, upper bound: 655.4342407
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334240, upper bound: 655.4334240
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334240, upper bound: 655.4334240
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4340507, upper bound: 655.4341474
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4340507, upper bound: 655.4340510
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339991, upper bound: 655.4340271
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339991, upper bound: 655.4340087
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323263, upper bound: 655.4322755
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323115, upper bound: 655.4322455
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4177529, upper bound: 655.4177529
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4177529, upper bound: 655.4177529
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308821, upper bound: 655.4308821
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308821, upper bound: 655.4308821
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
time: 0.69 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4333839, upper bound: 655.4333839
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4333839, upper bound: 655.4333839
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4310209
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4338577, upper bound: 655.4339844
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4338577, upper bound: 655.4338577
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4337458, upper bound: 655.4337786
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4337458, upper bound: 655.4337458
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4341260, upper bound: 655.4342747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4341259, upper bound: 655.4342407
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4334240, upper bound: 655.4334240
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4334240, upper bound: 655.4334240
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4340507, upper bound: 655.4341474
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4340507, upper bound: 655.4340510
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4339991, upper bound: 655.4340271
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4339991, upper bound: 655.4340087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4323263, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4323115, upper bound: 655.4322455
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4177529, upper bound: 655.4177529
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4177529, upper bound: 655.4177529
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4308821, upper bound: 655.4308821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4308821, upper bound: 655.4308821
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333114, upper bound: 655.4333114
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333114, upper bound: 655.4333114
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309106
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309111
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4339722
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4337573
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337457, upper bound: 655.4337458
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337457, upper bound: 655.4337786
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4342747
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4341213
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334232, upper bound: 655.4334232
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334232, upper bound: 655.4334232
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4095296, upper bound: 655.4095296
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4095296, upper bound: 655.4095296
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4336629
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320583, upper bound: 655.4320583
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320583, upper bound: 655.4320583
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323263, upper bound: 655.4322755
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322708, upper bound: 655.4322708
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322708, upper bound: 655.4322708
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320399, upper bound: 655.4320399
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320980, upper bound: 655.4320399
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309018, upper bound: 655.4309018
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309018, upper bound: 655.4309018
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321643, upper bound: 655.4321643
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321643, upper bound: 655.4321643
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4311235, upper bound: 655.4309403
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309993, upper bound: 655.4309403
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308447, upper bound: 655.4308436
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308437, upper bound: 655.4308436
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701
time: 0.67 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4333114, upper bound: 655.4333114
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4333114, upper bound: 655.4333114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309106
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309111
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4339722
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4337573
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4337457, upper bound: 655.4337458
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4337457, upper bound: 655.4337786
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4342747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4341213
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4334232, upper bound: 655.4334232
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4334232, upper bound: 655.4334232
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4095296, upper bound: 655.4095296
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4095296, upper bound: 655.4095296
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4336629
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4320583, upper bound: 655.4320583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4320583, upper bound: 655.4320583
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4323263, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322708, upper bound: 655.4322708
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322708, upper bound: 655.4322708
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4320399, upper bound: 655.4320399
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4320980, upper bound: 655.4320399
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309018, upper bound: 655.4309018
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309018, upper bound: 655.4309018
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4321643, upper bound: 655.4321643
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4321643, upper bound: 655.4321643
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4311235, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309993, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4308447, upper bound: 655.4308436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4308437, upper bound: 655.4308436
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333011, upper bound: 655.4333011
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333011, upper bound: 655.4333071
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332060, upper bound: 655.4332059
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332060, upper bound: 655.4332060
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320701, upper bound: 655.4320701
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320701, upper bound: 655.4320701
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309106
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4308518
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309111
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309060
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4339722
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4339678
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336097, upper bound: 655.4336097
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336097, upper bound: 655.4336097
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308691, upper bound: 655.4308691
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308691, upper bound: 655.4308691
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336048, upper bound: 655.4336048
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336048, upper bound: 655.4336047
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337346, upper bound: 655.4337753
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337346, upper bound: 655.4337346
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336367, upper bound: 655.4337531
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336367, upper bound: 655.4337827
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4341202
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4341213
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4308546
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4307854
time: 5.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4091687, upper bound: 655.4091687
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4091687, upper bound: 655.4091687
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335335, upper bound: 655.4335335
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335335, upper bound: 655.4335335
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335649, upper bound: 655.4336097
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4336629
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331047, upper bound: 655.4331047
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331047, upper bound: 655.4331046
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4167808, upper bound: 655.4167806
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4167808, upper bound: 655.4167806
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322445, upper bound: 655.4322445
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322445, upper bound: 655.4322445
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322416, upper bound: 655.4322416
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322416, upper bound: 655.4322416
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321424, upper bound: 655.4321424
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321424, upper bound: 655.4321424
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320029, upper bound: 655.4320029
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320029, upper bound: 655.4320029
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4318753, upper bound: 655.4318753
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4318753, upper bound: 655.4318753
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4319903, upper bound: 655.4319903
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4319903, upper bound: 655.4319903
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308506, upper bound: 655.4308506
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308506, upper bound: 655.4308506
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307940, upper bound: 655.4307940
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307940, upper bound: 655.4307940
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4322292, upper bound: 655.4321643
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321643, upper bound: 655.4321643
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4319230, upper bound: 655.4319230
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4319230, upper bound: 655.4319230
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4096053, upper bound: 655.4096053
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4096053, upper bound: 655.4096053
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4175093, upper bound: 655.4175093
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4175093, upper bound: 655.4175093
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307688, upper bound: 655.4307688
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307688, upper bound: 655.4307688
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4311235, upper bound: 655.4309403
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308291, upper bound: 655.4308291
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308291, upper bound: 655.4308291
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4305390, upper bound: 655.4305390
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4305390, upper bound: 655.4305390
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308437, upper bound: 655.4308437
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308448, upper bound: 655.4308436
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307513, upper bound: 655.4307513
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307513, upper bound: 655.4307513
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309095, upper bound: 655.4309095
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309095, upper bound: 655.4309095
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4305407, upper bound: 655.4305407
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306275, upper bound: 655.4305407
time: 0.70 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4333480, upper bound: 655.4333480
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4333011, upper bound: 655.4333011
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4333011, upper bound: 655.4333071
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4332060, upper bound: 655.4332059
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4332060, upper bound: 655.4332060
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4320701, upper bound: 655.4320701
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4320701, upper bound: 655.4320701
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309106
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4308518
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309111
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4309060
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4339722
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4337573, upper bound: 655.4339678
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4336097, upper bound: 655.4336097
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4336097, upper bound: 655.4336097
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308691, upper bound: 655.4308691
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308691, upper bound: 655.4308691
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4336048, upper bound: 655.4336048
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4336048, upper bound: 655.4336047
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4337346, upper bound: 655.4337753
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4337346, upper bound: 655.4337346
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4336367, upper bound: 655.4337531
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4336367, upper bound: 655.4337827
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4341202
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4341202, upper bound: 655.4341213
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4308546
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4307854
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4091687, upper bound: 655.4091687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4091687, upper bound: 655.4091687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4335335, upper bound: 655.4335335
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4335335, upper bound: 655.4335335
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4335649, upper bound: 655.4336097
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4336629
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4331047, upper bound: 655.4331047
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4331047, upper bound: 655.4331046
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4172617, upper bound: 655.4172617
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4167808, upper bound: 655.4167806
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4167808, upper bound: 655.4167806
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4317958, upper bound: 655.4317958
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322445, upper bound: 655.4322445
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322445, upper bound: 655.4322445
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322416, upper bound: 655.4322416
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322416, upper bound: 655.4322416
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4321424, upper bound: 655.4321424
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4321424, upper bound: 655.4321424
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4320029, upper bound: 655.4320029
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4320029, upper bound: 655.4320029
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4318753, upper bound: 655.4318753
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4318753, upper bound: 655.4318753
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322755, upper bound: 655.4322755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4174214, upper bound: 655.4174214
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4319903, upper bound: 655.4319903
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4319903, upper bound: 655.4319903
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322455, upper bound: 655.4322455
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322144, upper bound: 655.4322144
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308506, upper bound: 655.4308506
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308506, upper bound: 655.4308506
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307940, upper bound: 655.4307940
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307940, upper bound: 655.4307940
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4322292, upper bound: 655.4321643
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4321643, upper bound: 655.4321643
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4319230, upper bound: 655.4319230
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4319230, upper bound: 655.4319230
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4175325, upper bound: 655.4175325
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4096053, upper bound: 655.4096053
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4096053, upper bound: 655.4096053
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4175093, upper bound: 655.4175093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4175093, upper bound: 655.4175093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4176610, upper bound: 655.4176610
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307688, upper bound: 655.4307688
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307688, upper bound: 655.4307688
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4311235, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309403, upper bound: 655.4309403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308291, upper bound: 655.4308291
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308291, upper bound: 655.4308291
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307897, upper bound: 655.4307897
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4305390, upper bound: 655.4305390
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4305390, upper bound: 655.4305390
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308437, upper bound: 655.4308437
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4308448, upper bound: 655.4308436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307513, upper bound: 655.4307513
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4307513, upper bound: 655.4307513
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309467, upper bound: 655.4309467
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309095, upper bound: 655.4309095
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4309095, upper bound: 655.4309095
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4306701, upper bound: 655.4306701
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4305407, upper bound: 655.4305407
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 0, lower bound: -655.4306275, upper bound: 655.4305407

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333293, upper bound: 655.4333293
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333293, upper bound: 655.4333293
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4090247, upper bound: 655.4090247
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4090247, upper bound: 655.4090247
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4090247, upper bound: 655.4090247
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4090247, upper bound: 655.4090247
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332495, upper bound: 655.4332495
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332495, upper bound: 655.4332557
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307109, upper bound: 655.4307109
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307109, upper bound: 655.4307109
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330946, upper bound: 655.4330946
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330946, upper bound: 655.4330946
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4319763, upper bound: 655.4319763
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4319763, upper bound: 655.4319763
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320701, upper bound: 655.4320701
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320701, upper bound: 655.4320701
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307940, upper bound: 655.4308765
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307940, upper bound: 655.4307940
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4308518
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308518, upper bound: 655.4308518
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307043, upper bound: 655.4307707
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307043, upper bound: 655.4307476
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4305840, upper bound: 655.4306375
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4305840, upper bound: 655.4305840
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308297, upper bound: 655.4308297
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306547, upper bound: 655.4306547
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306547, upper bound: 655.4306547
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306547, upper bound: 655.4306547
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4306547, upper bound: 655.4306547
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092808, upper bound: 655.4093824
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092808, upper bound: 655.4093824
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337266, upper bound: 655.4337748
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337266, upper bound: 655.4338284
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334556, upper bound: 655.4334556
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334556, upper bound: 655.4334556
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4085805, upper bound: 655.4085805
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4085805, upper bound: 655.4085805
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307513, upper bound: 655.4307513
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307513, upper bound: 655.4307514
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308691, upper bound: 655.4308691
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4308691, upper bound: 655.4308691
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336048, upper bound: 655.4336048
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336048, upper bound: 655.4336047
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334959, upper bound: 655.4334959
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334959, upper bound: 655.4334959
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336991, upper bound: 655.4336991
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336991, upper bound: 655.4337218
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336991, upper bound: 655.4336991
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336991, upper bound: 655.4336991
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309373, upper bound: 655.4309373
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309373, upper bound: 655.4309373
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309228, upper bound: 655.4309228
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309427, upper bound: 655.4309427
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334843, upper bound: 655.4335980
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334843, upper bound: 655.4335980
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4329241, upper bound: 655.4329804
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4329241, upper bound: 655.4330499
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339911, upper bound: 655.4339911
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339911, upper bound: 655.4339911
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320284, upper bound: 655.4320284
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320284, upper bound: 655.4320284
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307659, upper bound: 655.4308415
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307659, upper bound: 655.4307659
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307517, upper bound: 655.4307543
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307517, upper bound: 655.4308698
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4307894
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307854, upper bound: 655.4309159
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4088332, upper bound: 655.4088332
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4088332, upper bound: 655.4088332
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309490, upper bound: 655.4309490
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307791, upper bound: 655.4307791
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307791, upper bound: 655.4307791
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307791, upper bound: 655.4307791
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4307791, upper bound: 655.4307791
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309057, upper bound: 655.4309057
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309057, upper bound: 655.4309057
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4091687, upper bound: 655.4091687
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4091687, upper bound: 655.4091687
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4091009, upper bound: 655.4091009
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4091009, upper bound: 655.4091009
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4092634, upper bound: 655.4092634
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335335, upper bound: 655.4335335
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335335, upper bound: 655.4335334
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334820, upper bound: 655.4334820
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334820, upper bound: 655.4334820
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331047, upper bound: 655.4331047
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331047, upper bound: 655.4331047
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4335650
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335650, upper bound: 655.4336342
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4326677, upper bound: 655.4326677
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4326677, upper bound: 655.4326677
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729
1: -107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362
2: -66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734
3: -134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476
4: -73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809

Time for backsubstitution: 1.04 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.74 + 417.31 = 420.05 seconds
