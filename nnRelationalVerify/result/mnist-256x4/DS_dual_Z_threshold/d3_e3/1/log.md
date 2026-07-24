## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.411746211


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716)
1: (-0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653)
2: (-0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282)
3: (-1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487)
4: (-1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829)
5: (-1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234)
6: (-1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365)
7: (-1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017)
8: (-1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066)
9: (-1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.30 + 3.77 = 6.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4856967, upper bound: 2.4856560
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4856560, upper bound: 2.4856967
time: 2.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.32
Output dim: 8, lower bound: -2.4856967, upper bound: 2.4856560
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.32
Output dim: 8, lower bound: -2.4856560, upper bound: 2.4856967

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4852067, upper bound: 2.4854399
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4855021, upper bound: 2.4851476
time: 1.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4851477, upper bound: 2.4855021
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4854399, upper bound: 2.4852067
time: 2.01 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.75 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 8, lower bound: -2.4852067, upper bound: 2.4854399
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 8, lower bound: -2.4855021, upper bound: 2.4851476
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 8, lower bound: -2.4851477, upper bound: 2.4855021
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 8, lower bound: -2.4854399, upper bound: 2.4852067

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677
time: 1.68 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.54 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.54
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728
time: 1.49 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.19
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.67 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.67 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523541
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523542
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523541
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523542
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
time: 1.84 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 6.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523541
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523542
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523541
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.10
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.10
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 6.07 + 596.43 = 602.50 seconds
