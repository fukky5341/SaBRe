## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.5895e-05


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261)
1: (-0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480)
2: (-0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418)
3: (-0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863)
4: (-0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.02 + 0.58 = 1.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000187, upper bound: 0.0000186

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000181, upper bound: 0.0000184
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000184, upper bound: 0.0000180
time: 0.15 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.33 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.33
Output dim: 0, lower bound: -0.0000181, upper bound: 0.0000184
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.33
Output dim: 0, lower bound: -0.0000184, upper bound: 0.0000180

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000184
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000173, upper bound: 0.0000183
time: 0.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000183, upper bound: 0.0000173
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000184, upper bound: 0.0000180
time: 0.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.41 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.41
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000184
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.41
Output dim: 0, lower bound: -0.0000173, upper bound: 0.0000183
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.41
Output dim: 0, lower bound: -0.0000183, upper bound: 0.0000173
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.41
Output dim: 0, lower bound: -0.0000184, upper bound: 0.0000180

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000175, upper bound: 0.0000184
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000179
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000181
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000171, upper bound: 0.0000170
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000171
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000170
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000178, upper bound: 0.0000179
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000170
time: 0.16 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.57 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000175, upper bound: 0.0000184
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000179
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000181
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000171, upper bound: 0.0000170
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000171
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000170
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000178, upper bound: 0.0000179
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.57
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000170

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000182
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000173, upper bound: 0.0000178
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000177
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000179, upper bound: 0.0000177
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000181
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000176
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000170
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000171, upper bound: 0.0000170
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000171
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000170
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000176, upper bound: 0.0000170
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000170
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000177, upper bound: 0.0000179
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000178, upper bound: 0.0000174
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000177, upper bound: 0.0000170
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000169
time: 0.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000182
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000173, upper bound: 0.0000178
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000177
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000179, upper bound: 0.0000177
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000181
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000176
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000170
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000171, upper bound: 0.0000170
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000171
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000170
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000176, upper bound: 0.0000170
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000170
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000177, upper bound: 0.0000179
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000178, upper bound: 0.0000174
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000177, upper bound: 0.0000170
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000182, upper bound: 0.0000169

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000177
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000181
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000172
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000176
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 22

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000155
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000153
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000164
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000164
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000170
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000176
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 34

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000154
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000152
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000164
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000165, upper bound: 0.0000164
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000154
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000149
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000153
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000151
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000154
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000152
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000154
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000151
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 34

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000162
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000162
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000154
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000154
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 22

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000177, upper bound: 0.0000170
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000172, upper bound: 0.0000170
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000154
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000152
time: 0.17 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000177
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000181
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000172
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000176
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000153
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000164
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000164
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000170
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000170, upper bound: 0.0000176
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000154
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000152
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000164
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000165, upper bound: 0.0000164
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000154
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000149
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000153
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000154
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000152
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000154
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000151
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000162
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000162
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000154
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000154
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000177, upper bound: 0.0000170
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000172, upper bound: 0.0000170
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000154
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.64
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000152

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000171
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000171
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000172
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000171
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000155
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000156
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000154
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000157
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000156
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000155
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000155
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000152
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000153
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000155
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000160
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000151
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000157
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000165
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000165
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000156
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000157
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000164
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000162
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000153
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000152
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000152
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000150
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000150
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000158
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000161
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000158
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000161
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000172, upper bound: 0.0000164
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000172, upper bound: 0.0000164
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000169, upper bound: 0.0000163
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000169, upper bound: 0.0000161
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000151
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000151
time: 0.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000171
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000171
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000172
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000171
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000173
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000156
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000154
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000156
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000152
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000153
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000155
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000160
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000151
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000165
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000156
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000154, upper bound: 0.0000157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000164
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000162
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000153
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000152
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000152
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000150
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000150
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000158
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000161
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000158
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000161
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000172, upper bound: 0.0000164
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000172, upper bound: 0.0000164
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000169, upper bound: 0.0000163
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000169, upper bound: 0.0000161
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000151
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000151

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000155
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000150
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000156
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000153
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000155
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000158
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000154
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000155
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000155
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000149
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000151
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000157
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000160
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000153
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000154
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000153
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000152
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000151
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000149
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000149
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000150
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000149
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000152
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000157
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000150
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000157
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000152
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000151
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000152
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000151
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000149
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000149
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000149
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000149
time: 0.16 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000150
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000156
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000153
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000158
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000154
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000155
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000149
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000151
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000151
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000160
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000153
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000154
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000153
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000152
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000151
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000149
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000149
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000150
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000149
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000152
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000157
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000150
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000157
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000152
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000151
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000152
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000151
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000153, upper bound: 0.0000149
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000149
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000149
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.41
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000149

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000159
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000159
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 24

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000147
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000148
time: 0.17 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.36 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000159
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000147
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000148

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000147, upper bound: 0.0000158
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000158
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000148
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000148
time: 0.16 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.42 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.42
Output dim: 0, lower bound: -0.0000147, upper bound: 0.0000158
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.42
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000158
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.42
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000148
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.42
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000148

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.60 + 112.65 = 114.25 seconds
