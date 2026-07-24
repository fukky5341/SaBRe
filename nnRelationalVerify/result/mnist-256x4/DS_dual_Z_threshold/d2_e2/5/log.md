## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.009087408


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0032582, 0.0039995, -0.0032582, 0.0039995, -0.0072577, 0.0072577)
1: (0.9874097, 1.0027788, 0.9874097, 1.0027788, -0.0153691, 0.0153691)
2: (-0.0097837, 0.0025387, -0.0097837, 0.0025387, -0.0123224, 0.0123224)
3: (-0.0013881, 0.0076918, -0.0013881, 0.0076918, -0.0090799, 0.0090799)
4: (-0.0035895, 0.0133376, -0.0035895, 0.0133376, -0.0169270, 0.0169270)
5: (-0.0043585, 0.0128399, -0.0043585, 0.0128399, -0.0171984, 0.0171984)
6: (-0.0092749, 0.0078574, -0.0092749, 0.0078574, -0.0171322, 0.0171322)
7: (-0.0107944, -0.0034377, -0.0107944, -0.0034377, -0.0073566, 0.0073566)
8: (-0.0050840, 0.0086255, -0.0050840, 0.0086255, -0.0137094, 0.0137094)
9: (-0.0078495, 0.0026556, -0.0078495, 0.0026556, -0.0105051, 0.0105051)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 3.09 = 4.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.55 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.55
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.55
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0032582, 0.0039995, -0.0032582, 0.0039995, -0.0072577, 0.0072577
1: 0.9874097, 1.0027788, 0.9874097, 1.0027788, -0.0153691, 0.0153691
2: -0.0097837, 0.0025387, -0.0097837, 0.0025387, -0.0123224, 0.0123224
3: -0.0013881, 0.0076918, -0.0013881, 0.0076918, -0.0090799, 0.0090799
4: -0.0035895, 0.0133376, -0.0035895, 0.0133376, -0.0169270, 0.0169270
5: -0.0043585, 0.0128399, -0.0043585, 0.0128399, -0.0171984, 0.0171984
6: -0.0092749, 0.0078574, -0.0092749, 0.0078574, -0.0171322, 0.0171322
7: -0.0107944, -0.0034377, -0.0107944, -0.0034377, -0.0073566, 0.0073566
8: -0.0050840, 0.0086255, -0.0050840, 0.0086255, -0.0137094, 0.0137094
9: -0.0078495, 0.0026556, -0.0078495, 0.0026556, -0.0105051, 0.0105051

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089464
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089465
time: 2.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0032582, 0.0039995, -0.0032582, 0.0039995, -0.0072577, 0.0072577
1: 0.9874097, 1.0027788, 0.9874097, 1.0027788, -0.0153691, 0.0153691
2: -0.0097837, 0.0025387, -0.0097837, 0.0025387, -0.0123224, 0.0123224
3: -0.0013881, 0.0076918, -0.0013881, 0.0076918, -0.0090799, 0.0090799
4: -0.0035895, 0.0133376, -0.0035895, 0.0133376, -0.0169270, 0.0169270
5: -0.0043585, 0.0128399, -0.0043585, 0.0128399, -0.0171984, 0.0171984
6: -0.0092749, 0.0078574, -0.0092749, 0.0078574, -0.0171322, 0.0171322
7: -0.0107944, -0.0034377, -0.0107944, -0.0034377, -0.0073566, 0.0073566
8: -0.0050840, 0.0086255, -0.0050840, 0.0086255, -0.0137094, 0.0137094
9: -0.0078495, 0.0026556, -0.0078495, 0.0026556, -0.0105051, 0.0105051

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089465
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089465
time: 2.03 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.91 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089464
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089465
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089465
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 1, lower bound: -0.0089465, upper bound: 0.0089465

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.70 + 16.04 = 20.74 seconds
