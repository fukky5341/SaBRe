## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.51975288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692)
1: (-0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081)
2: (-0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965)
3: (-0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550)
4: (-0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409)
5: (-0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894)
6: (-0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582)
7: (-0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213)
8: (-0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886)
9: (-0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.93 + 3.17 = 4.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5775032, upper bound: 0.5775021

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5627936, upper bound: 0.5630417
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5630417, upper bound: 0.5627925
time: 2.54 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.59
Output dim: 0, lower bound: -0.5627936, upper bound: 0.5630417
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.59
Output dim: 0, lower bound: -0.5630417, upper bound: 0.5627925

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5445144, upper bound: 0.5446745
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5445144, upper bound: 0.5446745
time: 1.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5590533, upper bound: 0.5588140
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5590620, upper bound: 0.5588109
time: 1.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 0, lower bound: -0.5445144, upper bound: 0.5446745
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 0, lower bound: -0.5445144, upper bound: 0.5446745
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 0, lower bound: -0.5590533, upper bound: 0.5588140
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 0, lower bound: -0.5590620, upper bound: 0.5588109

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5434764, upper bound: 0.5436294
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5434705, upper bound: 0.5436311
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 168

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5403019, upper bound: 0.5405047
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5403295, upper bound: 0.5404505
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566173, upper bound: 0.5564181
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566639, upper bound: 0.5564067
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5300579, upper bound: 0.5299454
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5300579, upper bound: 0.5299454
time: 1.63 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5434764, upper bound: 0.5436294
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5434705, upper bound: 0.5436311
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5403019, upper bound: 0.5405047
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5403295, upper bound: 0.5404505
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5566173, upper bound: 0.5564181
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5566639, upper bound: 0.5564067
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5300579, upper bound: 0.5299454
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -0.5300579, upper bound: 0.5299454

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231969, upper bound: 0.5233437
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231969, upper bound: 0.5233437
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5102551, upper bound: 0.5102540
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5102551, upper bound: 0.5102540
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366332, upper bound: 0.5368584
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366332, upper bound: 0.5368539
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5078284, upper bound: 0.5078799
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5078284, upper bound: 0.5078799
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5362899, upper bound: 0.5360998
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5362899, upper bound: 0.5360998
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5468773, upper bound: 0.5467205
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5469637, upper bound: 0.5466339
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5299656, upper bound: 0.5297406
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5298564, upper bound: 0.5298414
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5300578, upper bound: 0.5299229
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5300424, upper bound: 0.5299454
time: 1.76 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5231969, upper bound: 0.5233437
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5231969, upper bound: 0.5233437
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5102551, upper bound: 0.5102540
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5102551, upper bound: 0.5102540
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5366332, upper bound: 0.5368584
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5366332, upper bound: 0.5368539
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5078284, upper bound: 0.5078799
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5078284, upper bound: 0.5078799
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5362899, upper bound: 0.5360998
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5362899, upper bound: 0.5360998
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5468773, upper bound: 0.5467205
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5469637, upper bound: 0.5466339
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5299656, upper bound: 0.5297406
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5298564, upper bound: 0.5298414
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5300578, upper bound: 0.5299229
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.5300424, upper bound: 0.5299454

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4804759, upper bound: 0.4804622
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4804759, upper bound: 0.4804618
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5229671, upper bound: 0.5230611
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228924, upper bound: 0.5230914
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5042025, upper bound: 0.5042981
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5042025, upper bound: 0.5042981
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5357230, upper bound: 0.5359268
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5357177, upper bound: 0.5359283
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4796014, upper bound: 0.4796053
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4796014, upper bound: 0.4796053
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5013535, upper bound: 0.5013563
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5013535, upper bound: 0.5013563
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5461814, upper bound: 0.5460230
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5461637, upper bound: 0.5460283
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394612, upper bound: 0.5391490
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394612, upper bound: 0.5391478
time: 3.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5297636, upper bound: 0.5294922
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5296970, upper bound: 0.5295345
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5200348, upper bound: 0.5200399
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5200348, upper bound: 0.5200399
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5299886, upper bound: 0.5297779
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5299007, upper bound: 0.5298517
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5273458, upper bound: 0.5272644
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5273606, upper bound: 0.5272566
time: 1.82 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.4804759, upper bound: 0.4804622
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.4804759, upper bound: 0.4804618
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5229671, upper bound: 0.5230611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5228924, upper bound: 0.5230914
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5042025, upper bound: 0.5042981
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5042025, upper bound: 0.5042981
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5357230, upper bound: 0.5359268
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5357177, upper bound: 0.5359283
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.4796014, upper bound: 0.4796053
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.4796014, upper bound: 0.4796053
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5013535, upper bound: 0.5013563
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5013535, upper bound: 0.5013563
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5461814, upper bound: 0.5460230
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5461637, upper bound: 0.5460283
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5394612, upper bound: 0.5391490
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5394612, upper bound: 0.5391478
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5297636, upper bound: 0.5294922
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5296970, upper bound: 0.5295345
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5200348, upper bound: 0.5200399
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5200348, upper bound: 0.5200399
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5299886, upper bound: 0.5297779
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5299007, upper bound: 0.5298517
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5273458, upper bound: 0.5272644
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 0, lower bound: -0.5273606, upper bound: 0.5272566

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5229671, upper bound: 0.5230274
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5229431, upper bound: 0.5230611
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5087138, upper bound: 0.5089814
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5087734, upper bound: 0.5089312
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5355363, upper bound: 0.5356553
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354509, upper bound: 0.5357444
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5355285, upper bound: 0.5357019
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354880, upper bound: 0.5357642
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5460941, upper bound: 0.5458546
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5459380, upper bound: 0.5459295
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5162677, upper bound: 0.5162475
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5162677, upper bound: 0.5162475
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5095132, upper bound: 0.5093115
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5095132, upper bound: 0.5093115
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4962845, upper bound: 0.4962246
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4962845, upper bound: 0.4962246
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5296933, upper bound: 0.5293621
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5295858, upper bound: 0.5294206
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5269924, upper bound: 0.5268557
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5270186, upper bound: 0.5268481
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5173936, upper bound: 0.5174088
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5174026, upper bound: 0.5173990
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5200211, upper bound: 0.5200180
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199973, upper bound: 0.5200331
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5297319, upper bound: 0.5294704
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5296997, upper bound: 0.5295352
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271924, upper bound: 0.5271731
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5272172, upper bound: 0.5271654
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271203, upper bound: 0.5270350
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271201, upper bound: 0.5270373
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5176000, upper bound: 0.5175164
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5176000, upper bound: 0.5175164
time: 1.81 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5229671, upper bound: 0.5230274
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5229431, upper bound: 0.5230611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5087138, upper bound: 0.5089814
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5087734, upper bound: 0.5089312
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5355363, upper bound: 0.5356553
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5354509, upper bound: 0.5357444
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5355285, upper bound: 0.5357019
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5354880, upper bound: 0.5357642
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5460941, upper bound: 0.5458546
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5459380, upper bound: 0.5459295
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5162677, upper bound: 0.5162475
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5162677, upper bound: 0.5162475
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5095132, upper bound: 0.5093115
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5095132, upper bound: 0.5093115
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.4962845, upper bound: 0.4962246
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.4962845, upper bound: 0.4962246
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5296933, upper bound: 0.5293621
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5295858, upper bound: 0.5294206
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5269924, upper bound: 0.5268557
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5270186, upper bound: 0.5268481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5173936, upper bound: 0.5174088
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5174026, upper bound: 0.5173990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5200211, upper bound: 0.5200180
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5199973, upper bound: 0.5200331
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5297319, upper bound: 0.5294704
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5296997, upper bound: 0.5295352
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5271924, upper bound: 0.5271731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5272172, upper bound: 0.5271654
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5271203, upper bound: 0.5270350
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5271201, upper bound: 0.5270373
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5176000, upper bound: 0.5175164
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.45
Output dim: 0, lower bound: -0.5176000, upper bound: 0.5175164

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199196, upper bound: 0.5199944
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199229, upper bound: 0.5199855
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228904, upper bound: 0.5229583
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228546, upper bound: 0.5230134
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5031886, upper bound: 0.5031979
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5031886, upper bound: 0.5031969
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5145988, upper bound: 0.5148457
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5145988, upper bound: 0.5148457
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4646569, upper bound: 0.4646769
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4646569, upper bound: 0.4646769
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4838670, upper bound: 0.4839139
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4838670, upper bound: 0.4839139
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143918, upper bound: 0.5143070
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143918, upper bound: 0.5143070
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5161145, upper bound: 0.5161505
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5161145, upper bound: 0.5161505
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205528, upper bound: 0.5202488
time: 6.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205528, upper bound: 0.5202488
time: 7.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5293387, upper bound: 0.5291275
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5293056, upper bound: 0.5291743
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5268095, upper bound: 0.5266160
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5267515, upper bound: 0.5266688
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5176557, upper bound: 0.5175169
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5176557, upper bound: 0.5175169
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199454, upper bound: 0.5198177
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5198868, upper bound: 0.5199381
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5197335, upper bound: 0.5197226
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197152, upper bound: 0.5197934
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5270320, upper bound: 0.5267961
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5270486, upper bound: 0.5267846
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5083278, upper bound: 0.5081599
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5083278, upper bound: 0.5081599
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5180449, upper bound: 0.5180290
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5180449, upper bound: 0.5180301
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5269636, upper bound: 0.5268630
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5269312, upper bound: 0.5269278
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5175333, upper bound: 0.5174657
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5175333, upper bound: 0.5174657
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5270355, upper bound: 0.5268397
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5269365, upper bound: 0.5269336
time: 1.90 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5199196, upper bound: 0.5199944
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5199229, upper bound: 0.5199855
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5228904, upper bound: 0.5229583
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5228546, upper bound: 0.5230134
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5031886, upper bound: 0.5031979
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5031886, upper bound: 0.5031969
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5145988, upper bound: 0.5148457
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5145988, upper bound: 0.5148457
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.4646569, upper bound: 0.4646769
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.4646569, upper bound: 0.4646769
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.4838670, upper bound: 0.4839139
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.4838670, upper bound: 0.4839139
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5143918, upper bound: 0.5143070
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5143918, upper bound: 0.5143070
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5161145, upper bound: 0.5161505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5161145, upper bound: 0.5161505
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5205528, upper bound: 0.5202488
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5205528, upper bound: 0.5202488
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5293387, upper bound: 0.5291275
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5293056, upper bound: 0.5291743
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5268095, upper bound: 0.5266160
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5267515, upper bound: 0.5266688
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5176557, upper bound: 0.5175169
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5176557, upper bound: 0.5175169
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5199454, upper bound: 0.5198177
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5198868, upper bound: 0.5199381
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5197335, upper bound: 0.5197226
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5197152, upper bound: 0.5197934
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5270320, upper bound: 0.5267961
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5270486, upper bound: 0.5267846
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5083278, upper bound: 0.5081599
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5083278, upper bound: 0.5081599
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5180449, upper bound: 0.5180290
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5180449, upper bound: 0.5180301
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5269636, upper bound: 0.5268630
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5269312, upper bound: 0.5269278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5175333, upper bound: 0.5174657
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5175333, upper bound: 0.5174657
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5270355, upper bound: 0.5268397
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.37
Output dim: 0, lower bound: -0.5269365, upper bound: 0.5269336

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4774061, upper bound: 0.4773256
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4774061, upper bound: 0.4773246
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 168

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5154101, upper bound: 0.5154986
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5154317, upper bound: 0.5154502
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5195954, upper bound: 0.5196786
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5196069, upper bound: 0.5196565
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 168

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5183350, upper bound: 0.5185263
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5183513, upper bound: 0.5184702
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4895939, upper bound: 0.4894785
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4895939, upper bound: 0.4894785
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4705260, upper bound: 0.4704725
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4705260, upper bound: 0.4704725
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5202061, upper bound: 0.5200294
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5202061, upper bound: 0.5200294
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5292252, upper bound: 0.5289964
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291651, upper bound: 0.5291002
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5268095, upper bound: 0.5266050
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5268001, upper bound: 0.5266147
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4848913, upper bound: 0.4848915
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4848913, upper bound: 0.4848915
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197866, upper bound: 0.5196213
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5197318, upper bound: 0.5196435
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5172430, upper bound: 0.5173073
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5172496, upper bound: 0.5172969
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5101230, upper bound: 0.5102369
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5101571, upper bound: 0.5101909
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5249573, upper bound: 0.5247312
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5249573, upper bound: 0.5247325
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5269657, upper bound: 0.5265998
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5268686, upper bound: 0.5266934
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5178525, upper bound: 0.5178450
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179737, upper bound: 0.5177817
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5054637, upper bound: 0.5054611
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5054637, upper bound: 0.5054611
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264935, upper bound: 0.5262968
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264843, upper bound: 0.5262970
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5248979, upper bound: 0.5249198
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5248976, upper bound: 0.5249198
time: 1.66 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 8.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4774061, upper bound: 0.4773256
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4774061, upper bound: 0.4773246
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5154101, upper bound: 0.5154986
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5154317, upper bound: 0.5154502
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5195954, upper bound: 0.5196786
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5196069, upper bound: 0.5196565
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5183350, upper bound: 0.5185263
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5183513, upper bound: 0.5184702
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4895939, upper bound: 0.4894785
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4895939, upper bound: 0.4894785
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4705260, upper bound: 0.4704725
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4705260, upper bound: 0.4704725
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5202061, upper bound: 0.5200294
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5202061, upper bound: 0.5200294
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5292252, upper bound: 0.5289964
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5291651, upper bound: 0.5291002
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5268095, upper bound: 0.5266050
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5268001, upper bound: 0.5266147
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4848913, upper bound: 0.4848915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.4848913, upper bound: 0.4848915
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5197866, upper bound: 0.5196213
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5197318, upper bound: 0.5196435
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5172430, upper bound: 0.5173073
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5172496, upper bound: 0.5172969
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5101230, upper bound: 0.5102369
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5101571, upper bound: 0.5101909
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5249573, upper bound: 0.5247312
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5249573, upper bound: 0.5247325
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5269657, upper bound: 0.5265998
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5268686, upper bound: 0.5266934
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5178525, upper bound: 0.5178450
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5179737, upper bound: 0.5177817
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5054637, upper bound: 0.5054611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5054637, upper bound: 0.5054611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5264935, upper bound: 0.5262968
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5264843, upper bound: 0.5262970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5248979, upper bound: 0.5249198
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.20
Output dim: 0, lower bound: -0.5248976, upper bound: 0.5249198

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5172834, upper bound: 0.5171254
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5172995, upper bound: 0.5171146
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5111248, upper bound: 0.5110043
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5112309, upper bound: 0.5109556
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5200691, upper bound: 0.5199215
time: 8.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201847, upper bound: 0.5198660
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5290058, upper bound: 0.5288740
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5289584, upper bound: 0.5289408
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5265612, upper bound: 0.5263174
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5265351, upper bound: 0.5263779
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262568, upper bound: 0.5260699
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262501, upper bound: 0.5260706
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5113210, upper bound: 0.5111831
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5113210, upper bound: 0.5111831
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5247825, upper bound: 0.5244884
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5247347, upper bound: 0.5245670
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5012763, upper bound: 0.5011435
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5012763, upper bound: 0.5011435
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5267410, upper bound: 0.5263718
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5267331, upper bound: 0.5263718
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5268505, upper bound: 0.5266442
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5267790, upper bound: 0.5266794
time: 2.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5024172, upper bound: 0.5023029
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5024172, upper bound: 0.5023029
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5049956, upper bound: 0.5048477
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5049956, upper bound: 0.5048465
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5247751, upper bound: 0.5247184
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5246902, upper bound: 0.5247857
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5247879, upper bound: 0.5246930
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5247341, upper bound: 0.5248322
time: 2.21 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.85 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5172834, upper bound: 0.5171254
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5172995, upper bound: 0.5171146
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5111248, upper bound: 0.5110043
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5112309, upper bound: 0.5109556
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5200691, upper bound: 0.5199215
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5201847, upper bound: 0.5198660
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5290058, upper bound: 0.5288740
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5289584, upper bound: 0.5289408
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5265612, upper bound: 0.5263174
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5265351, upper bound: 0.5263779
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5262568, upper bound: 0.5260699
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5262501, upper bound: 0.5260706
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5113210, upper bound: 0.5111831
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5113210, upper bound: 0.5111831
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5247825, upper bound: 0.5244884
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5247347, upper bound: 0.5245670
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5012763, upper bound: 0.5011435
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5012763, upper bound: 0.5011435
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5267410, upper bound: 0.5263718
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5267331, upper bound: 0.5263718
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5268505, upper bound: 0.5266442
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5267790, upper bound: 0.5266794
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5024172, upper bound: 0.5023029
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5024172, upper bound: 0.5023029
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5049956, upper bound: 0.5048477
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5049956, upper bound: 0.5048465
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5247751, upper bound: 0.5247184
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5246902, upper bound: 0.5247857
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5247879, upper bound: 0.5246930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.85
Output dim: 0, lower bound: -0.5247341, upper bound: 0.5248322

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4774606, upper bound: 0.4774761
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4774606, upper bound: 0.4774761
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5200113, upper bound: 0.5196561
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199406, upper bound: 0.5196686
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5076717, upper bound: 0.5075311
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5076717, upper bound: 0.5075300
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5289584, upper bound: 0.5289336
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5289489, upper bound: 0.5289397
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264882, upper bound: 0.5261829
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264110, upper bound: 0.5262438
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264628, upper bound: 0.5262526
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5263809, upper bound: 0.5263004
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5020976, upper bound: 0.5019873
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5020976, upper bound: 0.5019873
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 168

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5215984, upper bound: 0.5214832
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5216483, upper bound: 0.5214019
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245990, upper bound: 0.5243096
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245912, upper bound: 0.5243119
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5241932, upper bound: 0.5240275
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5241847, upper bound: 0.5240283
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266720, upper bound: 0.5262287
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5265557, upper bound: 0.5262716
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5265579, upper bound: 0.5261393
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5265088, upper bound: 0.5262002
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4854093, upper bound: 0.4852525
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4854093, upper bound: 0.4852525
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5053897, upper bound: 0.5051987
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5053897, upper bound: 0.5051987
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157424, upper bound: 0.5156936
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157424, upper bound: 0.5156936
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5154280, upper bound: 0.5155450
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5154280, upper bound: 0.5155450
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5246622, upper bound: 0.5245000
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245770, upper bound: 0.5245583
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245945, upper bound: 0.5246099
time: 6.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245491, upper bound: 0.5246565
time: 2.08 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 11.02 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.4774606, upper bound: 0.4774761
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.4774606, upper bound: 0.4774761
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5200113, upper bound: 0.5196561
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5199406, upper bound: 0.5196686
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5076717, upper bound: 0.5075311
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5076717, upper bound: 0.5075300
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5289584, upper bound: 0.5289336
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5289489, upper bound: 0.5289397
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5264882, upper bound: 0.5261829
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5264110, upper bound: 0.5262438
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5264628, upper bound: 0.5262526
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5263809, upper bound: 0.5263004
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5020976, upper bound: 0.5019873
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5020976, upper bound: 0.5019873
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5215984, upper bound: 0.5214832
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5216483, upper bound: 0.5214019
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5245990, upper bound: 0.5243096
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5245912, upper bound: 0.5243119
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5241932, upper bound: 0.5240275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5241847, upper bound: 0.5240283
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5266720, upper bound: 0.5262287
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5265557, upper bound: 0.5262716
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5265579, upper bound: 0.5261393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5265088, upper bound: 0.5262002
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.4854093, upper bound: 0.4852525
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.4854093, upper bound: 0.4852525
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5053897, upper bound: 0.5051987
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5053897, upper bound: 0.5051987
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5157424, upper bound: 0.5156936
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5157424, upper bound: 0.5156936
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5154280, upper bound: 0.5155450
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5154280, upper bound: 0.5155450
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5246622, upper bound: 0.5245000
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5245770, upper bound: 0.5245583
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5245945, upper bound: 0.5246099
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 11.02
Output dim: 0, lower bound: -0.5245491, upper bound: 0.5246565

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4840938, upper bound: 0.4840184
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4840938, upper bound: 0.4840184
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5096715, upper bound: 0.5094533
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5096715, upper bound: 0.5094533
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4886587, upper bound: 0.4886450
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4886587, upper bound: 0.4886450
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692
1: -0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081
2: -0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965
3: -0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550
4: -0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409
5: -0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894
6: -0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582
7: -0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213
8: -0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886
9: -0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5050720, upper bound: 0.5051081
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5050720, upper bound: 0.5051081
time: 1.71 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 4.32 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.4840938, upper bound: 0.4840184
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.4840938, upper bound: 0.4840184
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.5096715, upper bound: 0.5094533
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.5096715, upper bound: 0.5094533
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.4886587, upper bound: 0.4886450
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.4886587, upper bound: 0.4886450
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.5050720, upper bound: 0.5051081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 4.32
Output dim: 0, lower bound: -0.5050720, upper bound: 0.5051081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5264882, upper bound: 0.5261829
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5264110, upper bound: 0.5262438
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5264628, upper bound: 0.5262526
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5263809, upper bound: 0.5263004
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5215984, upper bound: 0.5214832
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5216483, upper bound: 0.5214019
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5245990, upper bound: 0.5243096
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5245912, upper bound: 0.5243119
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5241932, upper bound: 0.5240275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5241847, upper bound: 0.5240283
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5266720, upper bound: 0.5262287
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5265557, upper bound: 0.5262716
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5265579, upper bound: 0.5261393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5265088, upper bound: 0.5262002
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5246622, upper bound: 0.5245000
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5245770, upper bound: 0.5245583
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5245945, upper bound: 0.5246099
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -0.5245491, upper bound: 0.5246565

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.11 + 596.22 = 600.33 seconds
