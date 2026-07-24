## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00262656


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063999, 0.0063999)
1: (-0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015947, 0.0015947)
2: (0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0084510, 0.0084510)
3: (-0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0038465, 0.0038465)
4: (0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016357, 0.0016357)
5: (0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0106292, 0.0106292)
6: (-0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026978, 0.0026978)
7: (-0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0069800, 0.0069800)
8: (-0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036707, 0.0036707)
9: (-0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0042564, 0.0042564)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 2.28 = 3.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0032832, upper bound: 0.0032832

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031713, upper bound: 0.0031002
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031002, upper bound: 0.0031713
time: 1.64 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.05 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.05
Output dim: 0, lower bound: -0.0031713, upper bound: 0.0031002
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.05
Output dim: 0, lower bound: -0.0031002, upper bound: 0.0031713

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063223, 0.0063138
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015753, 0.0015732
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083373, 0.0083485
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037999, 0.0037948
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016137, 0.0016158
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0104862, 0.0105002
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026651, 0.0026615
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068953, 0.0068861
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036262, 0.0036213
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041991, 0.0042047

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030721, upper bound: 0.0030275
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030853, upper bound: 0.0030123
time: 1.74 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063138, 0.0063223
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015732, 0.0015753
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083485, 0.0083373
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037948, 0.0037999
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016158, 0.0016137
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0105002, 0.0104862
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026615, 0.0026651
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068861, 0.0068953
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036213, 0.0036262
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0042047, 0.0041991

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030122, upper bound: 0.0030853
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030275, upper bound: 0.0030721
time: 1.67 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030721, upper bound: 0.0030275
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030853, upper bound: 0.0030123
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030122, upper bound: 0.0030853
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030275, upper bound: 0.0030721

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063087, 0.0063058
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015720, 0.0015712
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083268, 0.0083306
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037917, 0.0037900
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016116, 0.0016124
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0104729, 0.0104777
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026594, 0.0026581
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068806, 0.0068774
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036184, 0.0036168
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041938, 0.0041957

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025963, upper bound: 0.0025876
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025928, upper bound: 0.0025941
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063145, 0.0063003
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015734, 0.0015699
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083195, 0.0083382
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037952, 0.0037867
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016102, 0.0016138
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0104637, 0.0104872
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026618, 0.0026558
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068868, 0.0068714
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036217, 0.0036136
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041901, 0.0041996

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025963, upper bound: 0.0025876
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025928, upper bound: 0.0025941
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063003, 0.0063145
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015699, 0.0015734
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083382, 0.0083195
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037867, 0.0037952
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016138, 0.0016102
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0104872, 0.0104637
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026558, 0.0026618
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068714, 0.0068868
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036136, 0.0036217
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041996, 0.0041901

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025941, upper bound: 0.0025928
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025876, upper bound: 0.0025963
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063058, 0.0063087
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015712, 0.0015720
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083306, 0.0083268
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037900, 0.0037917
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016124, 0.0016116
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0104777, 0.0104729
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026581, 0.0026594
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068774, 0.0068806
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036168, 0.0036184
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041957, 0.0041938

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025941, upper bound: 0.0025928
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025876, upper bound: 0.0025963
time: 1.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.57 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025963, upper bound: 0.0025876
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025928, upper bound: 0.0025941
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025963, upper bound: 0.0025876
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025928, upper bound: 0.0025941
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025941, upper bound: 0.0025928
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025876, upper bound: 0.0025963
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025941, upper bound: 0.0025928
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 0, lower bound: -0.0025876, upper bound: 0.0025963

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.78 + 30.93 = 34.71 seconds
