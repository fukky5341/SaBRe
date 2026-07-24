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
execution time: IAR + RelationalAnalysis = 1.91 + 2.31 = 4.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4077811, upper bound: 0.4077880
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4077881, upper bound: 0.4077811
time: 1.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.85 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.85
Output dim: 8, lower bound: -0.4077811, upper bound: 0.4077880
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.85
Output dim: 8, lower bound: -0.4077881, upper bound: 0.4077811

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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3924602, upper bound: 0.3924704
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3924602, upper bound: 0.3924704
time: 1.28 seconds

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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3924704, upper bound: 0.3924602
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3924704, upper bound: 0.3924602
time: 1.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 8, lower bound: -0.3924602, upper bound: 0.3924704
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 8, lower bound: -0.3924602, upper bound: 0.3924704
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 8, lower bound: -0.3924704, upper bound: 0.3924602
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 8, lower bound: -0.3924704, upper bound: 0.3924602

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

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
time: 0.97 seconds

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

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
time: 0.91 seconds

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266
time: 1.13 seconds

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

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266
time: 1.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.60 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796267, upper bound: 0.3796305
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.60
Output dim: 8, lower bound: -0.3796305, upper bound: 0.3796266

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.22 + 27.73 = 31.95 seconds
