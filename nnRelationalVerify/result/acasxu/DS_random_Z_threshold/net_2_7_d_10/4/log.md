## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 65.1166706475


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772)
1: (-47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996)
2: (-25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529)
3: (-20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342)
4: (-31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.95 + 1.54 = 2.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -65.1818525, upper bound: 65.1818525

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1805473, upper bound: 65.1805473
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1805473, upper bound: 65.1815345
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 4, lower bound: -65.1805473, upper bound: 65.1805473
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 4, lower bound: -65.1805473, upper bound: 65.1815345

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1808858, upper bound: 65.1801589
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1807613, upper bound: 65.1801589
time: 0.46 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1004762, upper bound: 65.1008035
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1004762, upper bound: 65.1008035
time: 0.42 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.70 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.70
Output dim: 4, lower bound: -65.1808858, upper bound: 65.1801589
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.70
Output dim: 4, lower bound: -65.1807613, upper bound: 65.1801589
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.70
Output dim: 4, lower bound: -65.1004762, upper bound: 65.1008035
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 1.70
Output dim: 4, lower bound: -65.1004762, upper bound: 65.1008035

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1331210, upper bound: 65.1330565
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1331210, upper bound: 65.1330565
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1198645, upper bound: 65.1198645
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1198645, upper bound: 65.1198645
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.78
Output dim: 4, lower bound: -65.1331210, upper bound: 65.1330565
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.78
Output dim: 4, lower bound: -65.1331210, upper bound: 65.1330565
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.78
Output dim: 4, lower bound: -65.1198645, upper bound: 65.1198645
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.78
Output dim: 4, lower bound: -65.1198645, upper bound: 65.1198645

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1128036, upper bound: 65.1128036
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1128036, upper bound: 65.1128036
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1331210, upper bound: 65.1330565
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1330828, upper bound: 65.1330565
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1191023, upper bound: 65.1191023
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1191023, upper bound: 65.1191023
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1144157, upper bound: 65.1122595
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1143583, upper bound: 65.1122595
time: 0.48 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.86 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1128036, upper bound: 65.1128036
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1128036, upper bound: 65.1128036
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1331210, upper bound: 65.1330565
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1330828, upper bound: 65.1330565
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1191023, upper bound: 65.1191023
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1191023, upper bound: 65.1191023
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1144157, upper bound: 65.1122595
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.86
Output dim: 4, lower bound: -65.1143583, upper bound: 65.1122595

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1312064, upper bound: 65.1311199
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1330828, upper bound: 65.1330565
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1330565, upper bound: 65.1330565
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1136314, upper bound: 65.1114803
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1114803, upper bound: 65.1114803
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1312064, upper bound: 65.1311199
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1330828, upper bound: 65.1330565
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1330565, upper bound: 65.1330565
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1136314, upper bound: 65.1114803
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.78
Output dim: 4, lower bound: -65.1114803, upper bound: 65.1114803

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0834224, upper bound: 65.0834224
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0834224, upper bound: 65.0834224
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1312064, upper bound: 65.1311199
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0847524, upper bound: 65.0847524
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0847524, upper bound: 65.0847524
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1142721
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1142721
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197
time: 0.41 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.69 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.0834224, upper bound: 65.0834224
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.0834224, upper bound: 65.0834224
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.1312064, upper bound: 65.1311199
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.0847524, upper bound: 65.0847524
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.0847524, upper bound: 65.0847524
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1142721
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1142721
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.69
Output dim: 4, lower bound: -65.1189197, upper bound: 65.1189197

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560
time: 0.46 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.76 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -65.1141560, upper bound: 65.1141560

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
time: 0.42 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.70 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.70
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.70
Output dim: 4, lower bound: -65.1155700, upper bound: 65.1155700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.70
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.70
Output dim: 4, lower bound: -65.1311199, upper bound: 65.1311199

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282
time: 0.43 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.74 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.74
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.74
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.74
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.74
Output dim: 4, lower bound: -65.0607282, upper bound: 65.0607282

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.49 + 47.12 = 49.61 seconds
