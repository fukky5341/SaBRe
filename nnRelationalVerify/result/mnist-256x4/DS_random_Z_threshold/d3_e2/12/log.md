## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.38366993


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374)
1: (-0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201)
2: (-0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346)
3: (-0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890)
4: (-0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673)
5: (-0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257)
6: (-0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823)
7: (-0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595)
8: (0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006)
9: (-0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.88 + 2.18 = 3.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595
time: 0.98 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.07 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.07
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.07
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081556
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4081560, upper bound: 0.4081595
time: 1.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4081587, upper bound: 0.4081429
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4081428, upper bound: 0.4081587
time: 1.39 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081556
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 8, lower bound: -0.4081560, upper bound: 0.4081595
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 8, lower bound: -0.4081587, upper bound: 0.4081429
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 8, lower bound: -0.4081428, upper bound: 0.4081587

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4070295, upper bound: 0.4070141
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4070179, upper bound: 0.4070231
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3996177, upper bound: 0.3996196
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3996177, upper bound: 0.3996197
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4078482, upper bound: 0.4078332
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4078474, upper bound: 0.4078337
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4029051, upper bound: 0.4029176
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4029051, upper bound: 0.4029176
time: 1.00 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.4070295, upper bound: 0.4070141
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.4070179, upper bound: 0.4070231
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.3996177, upper bound: 0.3996196
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.3996177, upper bound: 0.3996197
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.4078482, upper bound: 0.4078332
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.4078474, upper bound: 0.4078337
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.4029051, upper bound: 0.4029176
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 8, lower bound: -0.4029051, upper bound: 0.4029176

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3907993, upper bound: 0.3907915
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3907993, upper bound: 0.3907915
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3924670, upper bound: 0.3924579
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3924670, upper bound: 0.3924579
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3907787, upper bound: 0.3907778
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3907787, upper bound: 0.3907778
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3992751, upper bound: 0.3992817
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3992751, upper bound: 0.3992817
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3722266, upper bound: 0.3722199
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3722266, upper bound: 0.3722199
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4025775, upper bound: 0.4025681
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4025775, upper bound: 0.4025681
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3956281, upper bound: 0.3956369
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3956281, upper bound: 0.3956369
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4025670, upper bound: 0.4025788
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4025654, upper bound: 0.4025793
time: 1.10 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3907993, upper bound: 0.3907915
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3907993, upper bound: 0.3907915
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3924670, upper bound: 0.3924579
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3924670, upper bound: 0.3924579
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3907787, upper bound: 0.3907778
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3907787, upper bound: 0.3907778
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3992751, upper bound: 0.3992817
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3992751, upper bound: 0.3992817
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3722266, upper bound: 0.3722199
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3722266, upper bound: 0.3722199
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.4025775, upper bound: 0.4025681
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.4025775, upper bound: 0.4025681
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3956281, upper bound: 0.3956369
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.3956281, upper bound: 0.3956369
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.4025670, upper bound: 0.4025788
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 8, lower bound: -0.4025654, upper bound: 0.4025793

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3790968, upper bound: 0.3790915
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3790968, upper bound: 0.3790915
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3729700, upper bound: 0.3729692
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3729700, upper bound: 0.3729692
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3917883, upper bound: 0.3917823
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3917885, upper bound: 0.3917793
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3820770, upper bound: 0.3820955
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3820969, upper bound: 0.3820735
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3907780, upper bound: 0.3907771
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3907780, upper bound: 0.3907771
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3897461, upper bound: 0.3897429
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3897402, upper bound: 0.3897454
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3951468, upper bound: 0.3951426
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3951468, upper bound: 0.3951426
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3932122, upper bound: 0.3932068
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3932122, upper bound: 0.3932068
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3943008, upper bound: 0.3943032
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3943008, upper bound: 0.3943032
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3867830, upper bound: 0.3867856
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3867830, upper bound: 0.3867856
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3951408, upper bound: 0.3951489
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3951408, upper bound: 0.3951489
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3758322, upper bound: 0.3758307
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3758322, upper bound: 0.3758307
time: 1.06 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3790968, upper bound: 0.3790915
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3790968, upper bound: 0.3790915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3729700, upper bound: 0.3729692
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3729700, upper bound: 0.3729692
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3917883, upper bound: 0.3917823
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3917885, upper bound: 0.3917793
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3820770, upper bound: 0.3820955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3820969, upper bound: 0.3820735
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3907780, upper bound: 0.3907771
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3907780, upper bound: 0.3907771
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3897461, upper bound: 0.3897429
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3897402, upper bound: 0.3897454
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3904222, upper bound: 0.3904258
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3951468, upper bound: 0.3951426
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3951468, upper bound: 0.3951426
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3932122, upper bound: 0.3932068
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3932122, upper bound: 0.3932068
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3943008, upper bound: 0.3943032
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3943008, upper bound: 0.3943032
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3867830, upper bound: 0.3867856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3867830, upper bound: 0.3867856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3951408, upper bound: 0.3951489
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3951408, upper bound: 0.3951489
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3758322, upper bound: 0.3758307
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 8, lower bound: -0.3758322, upper bound: 0.3758307

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3812035, upper bound: 0.3812239
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3812250, upper bound: 0.3812064
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3856008, upper bound: 0.3856050
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3856117, upper bound: 0.3855850
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3880621, upper bound: 0.3880591
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3880570, upper bound: 0.3880649
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3617777, upper bound: 0.3617838
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3617777, upper bound: 0.3617838
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3890039, upper bound: 0.3890124
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3890126, upper bound: 0.3889983
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3891742, upper bound: 0.3891838
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3891744, upper bound: 0.3891835
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574524
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574524
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574523
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574523
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3617267, upper bound: 0.3617283
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3617267, upper bound: 0.3617283
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3849212, upper bound: 0.3849286
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3849239, upper bound: 0.3849242
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3842096, upper bound: 0.3842084
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3842096, upper bound: 0.3842084
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3717677, upper bound: 0.3717657
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3717677, upper bound: 0.3717657
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3913868, upper bound: 0.3913853
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3913868, upper bound: 0.3913853
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3880988, upper bound: 0.3880952
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3881038, upper bound: 0.3880930
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3896971, upper bound: 0.3897061
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3897051, upper bound: 0.3897004
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3720969, upper bound: 0.3721001
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3720969, upper bound: 0.3721001
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3686700, upper bound: 0.3686826
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3686700, upper bound: 0.3686826
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3751432, upper bound: 0.3751477
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3751590, upper bound: 0.3751307
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3717620, upper bound: 0.3717713
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3717620, upper bound: 0.3717713
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3644301, upper bound: 0.3644330
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3644301, upper bound: 0.3644330
time: 0.98 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.79 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3812035, upper bound: 0.3812239
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3812250, upper bound: 0.3812064
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3856008, upper bound: 0.3856050
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3856117, upper bound: 0.3855850
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3880621, upper bound: 0.3880591
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3880570, upper bound: 0.3880649
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3617777, upper bound: 0.3617838
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3617777, upper bound: 0.3617838
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3890039, upper bound: 0.3890124
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3890126, upper bound: 0.3889983
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3891742, upper bound: 0.3891838
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3891744, upper bound: 0.3891835
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574524
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574524
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574523
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3574435, upper bound: 0.3574523
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3617267, upper bound: 0.3617283
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3617267, upper bound: 0.3617283
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3849212, upper bound: 0.3849286
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3849239, upper bound: 0.3849242
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3842096, upper bound: 0.3842084
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3842096, upper bound: 0.3842084
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3717677, upper bound: 0.3717657
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3717677, upper bound: 0.3717657
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3913868, upper bound: 0.3913853
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3913868, upper bound: 0.3913853
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3880988, upper bound: 0.3880952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3881038, upper bound: 0.3880930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3896971, upper bound: 0.3897061
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3897051, upper bound: 0.3897004
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3720969, upper bound: 0.3721001
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3720969, upper bound: 0.3721001
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3686700, upper bound: 0.3686826
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3686700, upper bound: 0.3686826
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3751432, upper bound: 0.3751477
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3751590, upper bound: 0.3751307
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3717620, upper bound: 0.3717713
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3717620, upper bound: 0.3717713
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3644301, upper bound: 0.3644330
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 8, lower bound: -0.3644301, upper bound: 0.3644330

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3754364, upper bound: 0.3754214
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3754364, upper bound: 0.3754214
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3847081, upper bound: 0.3846707
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3846923, upper bound: 0.3846809
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3757960, upper bound: 0.3758207
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3758278, upper bound: 0.3757921
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3777306, upper bound: 0.3777439
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3777306, upper bound: 0.3777439
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3639619, upper bound: 0.3639771
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3639619, upper bound: 0.3639771
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3890116, upper bound: 0.3889976
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3890119, upper bound: 0.3889976
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3825656, upper bound: 0.3825788
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3825656, upper bound: 0.3825788
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3845653, upper bound: 0.3845824
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3845692, upper bound: 0.3845803
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3318084, upper bound: 0.3318140
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3318084, upper bound: 0.3318140
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3739441, upper bound: 0.3739665
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3739678, upper bound: 0.3739341
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3764783, upper bound: 0.3764766
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3764783, upper bound: 0.3764766
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3548094, upper bound: 0.3548166
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3548094, upper bound: 0.3548166
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3437155, upper bound: 0.3437140
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3437155, upper bound: 0.3437140
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3902744, upper bound: 0.3902684
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3902707, upper bound: 0.3902740
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3790952, upper bound: 0.3790931
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3790952, upper bound: 0.3790931
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3660973, upper bound: 0.3660851
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3660973, upper bound: 0.3660851
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3891479, upper bound: 0.3891574
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3891471, upper bound: 0.3891573
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3746389, upper bound: 0.3746347
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3746389, upper bound: 0.3746347
time: 1.13 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.12 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3754364, upper bound: 0.3754214
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3754364, upper bound: 0.3754214
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3847081, upper bound: 0.3846707
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3846923, upper bound: 0.3846809
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3757960, upper bound: 0.3758207
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3758278, upper bound: 0.3757921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3777306, upper bound: 0.3777439
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3777306, upper bound: 0.3777439
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3639619, upper bound: 0.3639771
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3639619, upper bound: 0.3639771
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3890116, upper bound: 0.3889976
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3890119, upper bound: 0.3889976
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3825656, upper bound: 0.3825788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3825656, upper bound: 0.3825788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3845653, upper bound: 0.3845824
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3845692, upper bound: 0.3845803
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3318084, upper bound: 0.3318140
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3318084, upper bound: 0.3318140
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3739441, upper bound: 0.3739665
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3739678, upper bound: 0.3739341
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3764783, upper bound: 0.3764766
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3764783, upper bound: 0.3764766
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3548094, upper bound: 0.3548166
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3548094, upper bound: 0.3548166
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3437155, upper bound: 0.3437140
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3437155, upper bound: 0.3437140
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3902744, upper bound: 0.3902684
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3902707, upper bound: 0.3902740
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3790952, upper bound: 0.3790931
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3790952, upper bound: 0.3790931
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3660973, upper bound: 0.3660851
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3660973, upper bound: 0.3660851
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3891479, upper bound: 0.3891574
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3891471, upper bound: 0.3891573
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3746389, upper bound: 0.3746347
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 8, lower bound: -0.3746389, upper bound: 0.3746347

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3749255, upper bound: 0.3749034
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3749255, upper bound: 0.3749034
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3746933, upper bound: 0.3746604
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3746933, upper bound: 0.3746604
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3807147, upper bound: 0.3807171
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3807226, upper bound: 0.3807091
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3862929, upper bound: 0.3862696
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3862883, upper bound: 0.3862772
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3775835, upper bound: 0.3776021
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3775835, upper bound: 0.3776021
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3477781, upper bound: 0.3477884
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3477781, upper bound: 0.3477884
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3902744, upper bound: 0.3902460
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3902594, upper bound: 0.3902683
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3804681, upper bound: 0.3804691
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3804681, upper bound: 0.3804691
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3771291, upper bound: 0.3771346
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3771291, upper bound: 0.3771346
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3877499, upper bound: 0.3877506
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3877398, upper bound: 0.3877607
time: 1.10 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.76 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3749255, upper bound: 0.3749034
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3749255, upper bound: 0.3749034
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3746933, upper bound: 0.3746604
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3746933, upper bound: 0.3746604
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3807147, upper bound: 0.3807171
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3807226, upper bound: 0.3807091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3862929, upper bound: 0.3862696
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3862883, upper bound: 0.3862772
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3775835, upper bound: 0.3776021
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3775835, upper bound: 0.3776021
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3477781, upper bound: 0.3477884
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3477781, upper bound: 0.3477884
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3902744, upper bound: 0.3902460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3902594, upper bound: 0.3902683
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3804681, upper bound: 0.3804691
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3804681, upper bound: 0.3804691
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3771291, upper bound: 0.3771346
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3771291, upper bound: 0.3771346
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3877499, upper bound: 0.3877506
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 8, lower bound: -0.3877398, upper bound: 0.3877607

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3611139, upper bound: 0.3610930
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3611139, upper bound: 0.3610930
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3798357, upper bound: 0.3798345
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3798357, upper bound: 0.3798345
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3895511, upper bound: 0.3895272
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3895551, upper bound: 0.3895228
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3418888, upper bound: 0.3418951
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3418888, upper bound: 0.3418951
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3827309, upper bound: 0.3827333
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3827309, upper bound: 0.3827333
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3786628, upper bound: 0.3786854
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3786722, upper bound: 0.3786760
time: 1.22 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3611139, upper bound: 0.3610930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3611139, upper bound: 0.3610930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3798357, upper bound: 0.3798345
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3798357, upper bound: 0.3798345
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3895511, upper bound: 0.3895272
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3895551, upper bound: 0.3895228
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3418888, upper bound: 0.3418951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3418888, upper bound: 0.3418951
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3827309, upper bound: 0.3827333
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3827309, upper bound: 0.3827333
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3786628, upper bound: 0.3786854
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 8, lower bound: -0.3786722, upper bound: 0.3786760

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3601941, upper bound: 0.3601834
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3601941, upper bound: 0.3601834
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374
1: -0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201
2: -0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346
3: -0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890
4: -0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673
5: -0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257
6: -0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823
7: -0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595
8: 0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006
9: -0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3690005, upper bound: 0.3689691
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3690005, upper bound: 0.3689691
time: 1.18 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 3.22 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.22
Output dim: 8, lower bound: -0.3601941, upper bound: 0.3601834
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.22
Output dim: 8, lower bound: -0.3601941, upper bound: 0.3601834
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.22
Output dim: 8, lower bound: -0.3690005, upper bound: 0.3689691
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.22
Output dim: 8, lower bound: -0.3690005, upper bound: 0.3689691

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.06 + 263.67 = 266.73 seconds
