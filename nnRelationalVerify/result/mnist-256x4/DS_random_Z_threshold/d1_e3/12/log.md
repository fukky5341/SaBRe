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
execution time: IAR + RelationalAnalysis = 1.01 + 2.12 = 3.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.73 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.81 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.81
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.81
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0284633, upper bound: 0.0284633
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0284633, upper bound: 0.0284633
time: 0.91 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0239461, upper bound: 0.0239461
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0239461, upper bound: 0.0239461
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.06 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 7, lower bound: -0.0284633, upper bound: 0.0284633
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 7, lower bound: -0.0284633, upper bound: 0.0284633
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.06
Output dim: 7, lower bound: -0.0239461, upper bound: 0.0239461
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.06
Output dim: 7, lower bound: -0.0239461, upper bound: 0.0239461

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
time: 0.80 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0276732, upper bound: 0.0276732
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0276732, upper bound: 0.0276732
time: 1.08 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.08 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0276732, upper bound: 0.0276732
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 7, lower bound: -0.0276732, upper bound: 0.0276732

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
time: 0.67 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
time: 0.74 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0257096, upper bound: 0.0257096
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0257096, upper bound: 0.0257096
time: 0.78 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0253713, upper bound: 0.0253713
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0253713, upper bound: 0.0253713
time: 0.81 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0266560, upper bound: 0.0266560
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0257096, upper bound: 0.0257096
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0257096, upper bound: 0.0257096
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0253713, upper bound: 0.0253713
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 7, lower bound: -0.0253713, upper bound: 0.0253713

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
time: 0.67 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0233900, upper bound: 0.0233900
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0233900, upper bound: 0.0233900
time: 0.62 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.91 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 7, lower bound: -0.0250999, upper bound: 0.0250999
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 7, lower bound: -0.0233900, upper bound: 0.0233900
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 7, lower bound: -0.0233900, upper bound: 0.0233900

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.12 + 34.29 = 37.41 seconds
