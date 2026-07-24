## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.025922619999999997


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566)
1: (-0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010)
2: (0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271)
3: (-0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418)
4: (-0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234)
5: (0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077)
6: (-0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918)
7: (0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359)
8: (-0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140)
9: (-0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 2.12 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.73 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.95
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.95
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988
time: 1.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988
time: 1.30 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 7, lower bound: -0.0291988, upper bound: 0.0291988

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.08 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0271150, upper bound: 0.0271150

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
time: 0.92 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 7, lower bound: -0.0269362, upper bound: 0.0269362

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.99 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
time: 1.35 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 7, lower bound: -0.0261010, upper bound: 0.0261010

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566
1: -0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010
2: 0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271
3: -0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418
4: -0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234
5: 0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077
6: -0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918
7: 0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359
8: -0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140
9: -0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
time: 0.99 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 7, lower bound: -0.0247977, upper bound: 0.0247977

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.40 + 472.00 = 475.41 seconds
