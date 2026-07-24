## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.019857408


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372)
1: (-0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658)
2: (0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212)
3: (-0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573)
4: (0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.72 + 0.75 = 3.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0206848

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205565, upper bound: 0.0206848
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0205565
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.81 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.0205565, upper bound: 0.0206848
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0205565

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 2.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196732, upper bound: 0.0197263
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196732, upper bound: 0.0197263
time: 0.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 2.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0197263, upper bound: 0.0196732
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0197263, upper bound: 0.0196732
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.31 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.31
Output dim: 0, lower bound: -0.0196732, upper bound: 0.0197263
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.31
Output dim: 0, lower bound: -0.0196732, upper bound: 0.0197263
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.31
Output dim: 0, lower bound: -0.0197263, upper bound: 0.0196732
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.31
Output dim: 0, lower bound: -0.0197263, upper bound: 0.0196732

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.47 + 7.43 = 10.91 seconds
