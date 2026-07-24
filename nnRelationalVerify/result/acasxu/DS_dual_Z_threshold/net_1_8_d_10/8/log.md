## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 20.60317678965


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7843938, 3.3221977, -3.7843938, 3.3221977, -7.1065907, 7.1065907)
1: (-14.9787064, 12.8845959, -14.9787064, 12.8845959, -27.8633022, 27.8633022)
2: (-7.4894867, 12.0534105, -7.4894867, 12.0534105, -19.5428963, 19.5428963)
3: (-13.1016846, 11.7157326, -13.1016846, 11.7157326, -24.8174152, 24.8174152)
4: (-9.5921707, 12.2164268, -9.5921707, 12.2164268, -21.8085976, 21.8085976)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.06 + 1.92 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -20.6042070, upper bound: 20.6042070

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026400
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026400
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.38 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 1.38
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026400
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 1.38
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026400

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.98 + 1.38 = 4.36 seconds
