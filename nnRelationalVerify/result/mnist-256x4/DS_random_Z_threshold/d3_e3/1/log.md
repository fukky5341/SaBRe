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
execution time: IAR + RelationalAnalysis = 0.89 + 3.64 = 4.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377
time: 2.28 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.76 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.76
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.76
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3709284, upper bound: 2.3709282
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3709284, upper bound: 2.3709282
time: 1.39 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4863291, upper bound: 2.4863377
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863292
time: 2.05 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.82 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 5.82
Output dim: 8, lower bound: -2.3709284, upper bound: 2.3709282
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 5.82
Output dim: 8, lower bound: -2.3709284, upper bound: 2.3709282
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.82
Output dim: 8, lower bound: -2.4863291, upper bound: 2.4863377
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.82
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863292

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4862234, upper bound: 2.4862709
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4862603, upper bound: 2.4862289
time: 1.84 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4789280, upper bound: 2.4791351
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4791949, upper bound: 2.4788668
time: 1.69 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.36 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.36
Output dim: 8, lower bound: -2.4862234, upper bound: 2.4862709
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.36
Output dim: 8, lower bound: -2.4862603, upper bound: 2.4862289
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.36
Output dim: 8, lower bound: -2.4789280, upper bound: 2.4791351
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.36
Output dim: 8, lower bound: -2.4791949, upper bound: 2.4788668

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.2455913, upper bound: 2.2456077
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.2455913, upper bound: 2.2456077
time: 1.07 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4862598, upper bound: 2.4862289
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4862605, upper bound: 2.4862279
time: 1.58 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.1486677, upper bound: 2.1486791
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.1486677, upper bound: 2.1486791
time: 1.21 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3885277, upper bound: 2.3884515
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3885277, upper bound: 2.3884515
time: 1.35 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.65 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.2455913, upper bound: 2.2456077
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.2455913, upper bound: 2.2456077
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.4862598, upper bound: 2.4862289
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.4862605, upper bound: 2.4862279
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.1486677, upper bound: 2.1486791
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.1486677, upper bound: 2.1486791
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.3885277, upper bound: 2.3884515
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.65
Output dim: 8, lower bound: -2.3885277, upper bound: 2.3884515

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4157461, upper bound: 2.4156901
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4157461, upper bound: 2.4156901
time: 1.25 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3801142, upper bound: 2.3800585
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3801142, upper bound: 2.3800585
time: 1.33 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.50 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 8, lower bound: -2.4157461, upper bound: 2.4156901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 8, lower bound: -2.4157461, upper bound: 2.4156901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 8, lower bound: -2.3801142, upper bound: 2.3800585
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 8, lower bound: -2.3801142, upper bound: 2.3800585

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3904959, upper bound: 2.3904311
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3904959, upper bound: 2.3904311
time: 1.50 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3881567, upper bound: 2.3881456
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3881963, upper bound: 2.3881220
time: 1.44 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.68 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.68
Output dim: 8, lower bound: -2.3904959, upper bound: 2.3904311
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.68
Output dim: 8, lower bound: -2.3904959, upper bound: 2.3904311
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.68
Output dim: 8, lower bound: -2.3881567, upper bound: 2.3881456
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.68
Output dim: 8, lower bound: -2.3881963, upper bound: 2.3881220

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.53 + 51.99 = 56.52 seconds
