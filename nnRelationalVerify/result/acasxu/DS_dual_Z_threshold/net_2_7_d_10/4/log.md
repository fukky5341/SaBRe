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
execution time: IAR + RelationalAnalysis = 2.55 + 1.71 = 4.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -65.1818525, upper bound: 65.1818525

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1746748, upper bound: 65.1746748
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1746748, upper bound: 65.1769633
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.48 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 4, lower bound: -65.1746748, upper bound: 65.1746748
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 4, lower bound: -65.1746748, upper bound: 65.1769633

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1745969, upper bound: 65.1745969
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1759053, upper bound: 65.1745969
time: 0.61 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1745969, upper bound: 65.1759053
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1745969, upper bound: 65.1769590
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.75 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 4, lower bound: -65.1745969, upper bound: 65.1745969
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 4, lower bound: -65.1759053, upper bound: 65.1745969
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 4, lower bound: -65.1745969, upper bound: 65.1759053
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 4, lower bound: -65.1745969, upper bound: 65.1769590

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1227562, upper bound: 65.1183563
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.75 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1184056, upper bound: 65.1183563
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1227562, upper bound: 65.1183563
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1227562
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1183563
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -65.1183563, upper bound: 65.1184056

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0854076, upper bound: 65.0846683
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0854076, upper bound: 65.0846683
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0855228, upper bound: 65.0846683
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0855228, upper bound: 65.0846683
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0855228
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0855228
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0854076
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0854076
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772
1: -47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996
2: -25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529
3: -20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342
4: -31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322
time: 0.53 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0847322, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0854076, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0854076, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0855228, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0855228, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0855228
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0855228
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0854076
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0854076
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0846683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.71
Output dim: 4, lower bound: -65.0846683, upper bound: 65.0847322

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.26 + 113.31 = 117.57 seconds
