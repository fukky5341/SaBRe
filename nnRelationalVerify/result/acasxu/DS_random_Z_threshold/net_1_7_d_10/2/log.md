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
execution time: IAR + RelationalAnalysis = 0.97 + 0.57 = 1.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000759
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000760
time: 0.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.29 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 0.29
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000759
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 0.29
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000760

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.54 + 0.29 = 1.83 seconds
