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
execution time: IAR + RelationalAnalysis = 1.55 + 2.20 = 3.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -110.3029335, upper bound: 110.3029335

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2817471, upper bound: 110.2977862
time: 1.02 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.98 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.98
Output dim: 3, lower bound: -110.2817471, upper bound: 110.2977862
NS_A2, status: Status.VERIFIED, split count: 1, time: 1.98
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -44.0261078, 64.2232971, -49.3401833, 71.9582520, -115.9843597, 113.5634766
1: -35.3125572, 59.3043060, -39.4408836, 66.3552780, -101.6678238, 98.7451935
2: -29.9509239, 62.9027901, -33.4991455, 70.2209854, -100.1719055, 96.4019241
3: -48.1699867, 60.4233284, -53.5254898, 67.6930313, -115.8630219, 113.9488220
4: -34.6387939, 67.9170990, -38.8239403, 75.6963577, -110.3351440, 106.7410355

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108
time: 1.03 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108
time: 0.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.45 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.45
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.45
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.76 + 5.43 = 9.19 seconds
