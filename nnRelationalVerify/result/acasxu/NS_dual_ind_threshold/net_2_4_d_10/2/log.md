## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.07289891999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720)
1: (-0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452)
2: (0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376)
3: (-0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086)
4: (0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.94 + 0.67 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0809988, upper bound: 0.0809988

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0709466, upper bound: 0.0793979
time: 0.23 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0698267, upper bound: 0.0698267
time: 0.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.61 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0709466, upper bound: 0.0793979
NS_A2, status: Status.VERIFIED, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0698267, upper bound: 0.0698267

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0316362, 0.1329694, 0.0313130, 0.1384850, -0.1068487, 0.1016564
1: -0.0140712, 0.0123975, -0.0147149, 0.0132303, -0.0273015, 0.0271123
2: 0.0274415, 0.0525841, 0.0269715, 0.0536092, -0.0261676, 0.0256126
3: -0.0196888, -0.0021840, -0.0199371, -0.0015285, -0.0181603, 0.0177531
4: 0.0238663, 0.0473348, 0.0235005, 0.0482717, -0.0244053, 0.0238343

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0698267, upper bound: 0.0698267
time: 0.23 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0698267, upper bound: 0.0698267
time: 0.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.39 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0698267, upper bound: 0.0698267
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0698267, upper bound: 0.0698267

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 2.60 + 3.00 = 5.60 seconds
