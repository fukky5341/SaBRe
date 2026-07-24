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
execution time: IAR + RelationalAnalysis = 1.35 + 0.98 = 2.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.8699125, upper bound: 1.8699125

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8692398, upper bound: 1.8502497
time: 0.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806
time: 0.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.70 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.8692398, upper bound: 1.8502497
NS_A2, status: Status.VERIFIED, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.4915886, 0.5761790, -1.5446405, 0.9670887, -1.4586773, 2.1208196
1: -0.7926064, 0.5389530, -1.4420700, 0.8331291, -1.6257355, 1.9810231
2: -0.5954361, 0.5697228, -1.1730814, 0.8038781, -1.3993142, 1.7428043
3: -0.8339477, 0.5731941, -1.4580498, 0.8813943, -1.7153419, 2.0312438
4: -0.6962419, 0.6630035, -1.3076820, 0.9560763, -1.6523181, 1.9706855

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806
time: 0.26 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.92 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 1.92
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 1.92
Output dim: 0, lower bound: -1.8499806, upper bound: 1.8499806

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 2.33 + 2.62 = 4.95 seconds
