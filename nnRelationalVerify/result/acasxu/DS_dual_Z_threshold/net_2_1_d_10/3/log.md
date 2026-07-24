## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.8512133750000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292)
1: (-1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992)
2: (-1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595)
3: (-1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442)
4: (-1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 0.98 = 2.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.8699125, upper bound: 1.8699125

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.71 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 0.71
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 0.71
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.23 + 0.71 = 2.94 seconds
