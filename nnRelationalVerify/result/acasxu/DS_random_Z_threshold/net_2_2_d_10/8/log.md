## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 110.29190320664999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-49.3401833, 71.9582520, -49.3401833, 71.9582520, -121.2984314, 121.2984314)
1: (-39.4408836, 66.3552780, -39.4408836, 66.3552780, -105.7961578, 105.7961578)
2: (-33.4991455, 70.2209854, -33.4991455, 70.2209854, -103.7201309, 103.7201309)
3: (-53.5254898, 67.6930313, -53.5254898, 67.6930313, -121.2185211, 121.2185211)
4: (-38.8239403, 75.6963577, -38.8239403, 75.6963577, -114.5202942, 114.5202942)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.63 + 2.14 = 2.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -110.3029335, upper bound: 110.3029335

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.3020073, upper bound: 110.3021980
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.3021980, upper bound: 110.3020073
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.58 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -110.3020073, upper bound: 110.3021980
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -110.3021980, upper bound: 110.3020073

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -49.3401833, 71.9582520, -49.3401833, 71.9582520, -121.2984314, 121.2984314
1: -39.4408836, 66.3552780, -39.4408836, 66.3552780, -105.7961578, 105.7961578
2: -33.4991455, 70.2209854, -33.4991455, 70.2209854, -103.7201309, 103.7201309
3: -53.5254898, 67.6930313, -53.5254898, 67.6930313, -121.2185211, 121.2185211
4: -38.8239403, 75.6963577, -38.8239403, 75.6963577, -114.5202942, 114.5202942

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2736219, upper bound: 110.2736893
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2736219, upper bound: 110.2736893
time: 0.66 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -49.3401833, 71.9582520, -49.3401833, 71.9582520, -121.2984314, 121.2984314
1: -39.4408836, 66.3552780, -39.4408836, 66.3552780, -105.7961578, 105.7961578
2: -33.4991455, 70.2209854, -33.4991455, 70.2209854, -103.7201309, 103.7201309
3: -53.5254898, 67.6930313, -53.5254898, 67.6930313, -121.2185211, 121.2185211
4: -38.8239403, 75.6963577, -38.8239403, 75.6963577, -114.5202942, 114.5202942

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2912401, upper bound: 110.2912401
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2912401, upper bound: 110.2912401
time: 0.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.02
Output dim: 3, lower bound: -110.2736219, upper bound: 110.2736893
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.02
Output dim: 3, lower bound: -110.2736219, upper bound: 110.2736893
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.02
Output dim: 3, lower bound: -110.2912401, upper bound: 110.2912401
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.02
Output dim: 3, lower bound: -110.2912401, upper bound: 110.2912401

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.76 + 5.48 = 8.24 seconds
