## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 3084.599462796909


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1569.5086670, 2773.5766602, -1569.5086670, 2773.5766602, -4343.0854492, 4343.0854492)
1: (-490.6097412, 1099.5386963, -490.6097412, 1099.5386963, -1590.1484375, 1590.1484375)
2: (-307.9147949, 1094.9610596, -307.9147949, 1094.9610596, -1402.8758545, 1402.8758545)
3: (-652.4367676, 1308.2292480, -652.4367676, 1308.2292480, -1960.6660156, 1960.6660156)
4: (-339.8120422, 1133.2332764, -339.8120422, 1133.2332764, -1473.0451660, 1473.0451660)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.11 + 2.35 = 4.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3084.6303091, upper bound: 3084.6303091

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3084.5772438, upper bound: 3084.5549987
time: 1.13 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3084.5432863, upper bound: 3084.5432863
time: 0.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.28 seconds
NS_B1, status: Status.VERIFIED, split count: 1, time: 2.28
Output dim: 0, lower bound: -3084.5772438, upper bound: 3084.5549987
NS_B2, status: Status.VERIFIED, split count: 1, time: 2.28
Output dim: 0, lower bound: -3084.5432863, upper bound: 3084.5432863

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.46 + 2.28 = 6.73 seconds
