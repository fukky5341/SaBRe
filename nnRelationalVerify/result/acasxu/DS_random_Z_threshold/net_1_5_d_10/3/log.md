## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.045175422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553)
1: (-0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822)
2: (-0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808)
3: (-0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506)
4: (-0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.94 + 0.79 = 1.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0465726, upper bound: 0.0465726

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462716
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462716, upper bound: 0.0462715
time: 0.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462716
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0462716, upper bound: 0.0462715

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
time: 0.20 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
time: 0.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.49 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.49
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 1.49
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.49
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 1.49
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.73 + 3.31 = 5.05 seconds
