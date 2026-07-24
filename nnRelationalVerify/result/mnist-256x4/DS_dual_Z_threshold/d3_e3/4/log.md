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
execution time: IAR + RelationalAnalysis = 2.22 + 3.33 = 5.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5775032, upper bound: 0.5775021

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407550, upper bound: 0.5407550
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407550, upper bound: 0.5407550
time: 1.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.95
Output dim: 0, lower bound: -0.5407550, upper bound: 0.5407550
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.95
Output dim: 0, lower bound: -0.5407550, upper bound: 0.5407550

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5405808, upper bound: 0.5405704
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5405704, upper bound: 0.5405798
time: 1.60 seconds

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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5405808, upper bound: 0.5405704
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5405704, upper bound: 0.5405798
time: 1.58 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 7.21 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.21
Output dim: 0, lower bound: -0.5405808, upper bound: 0.5405704
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.21
Output dim: 0, lower bound: -0.5405704, upper bound: 0.5405798
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.21
Output dim: 0, lower bound: -0.5405808, upper bound: 0.5405704
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.21
Output dim: 0, lower bound: -0.5405704, upper bound: 0.5405798

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

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
time: 1.33 seconds

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

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217
time: 1.41 seconds

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

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
time: 1.33 seconds

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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217
time: 1.43 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.11 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179217, upper bound: 0.5179093
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.11
Output dim: 0, lower bound: -0.5179103, upper bound: 0.5179217

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.56 + 37.38 = 42.94 seconds
