## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01340901


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916)
1: (-0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314)
2: (0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552)
3: (-0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636)
4: (-0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996)
5: (-0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761)
6: (-0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070595, 0.0070595)
7: (-0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231)
8: (-0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160)
9: (0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.78 + 2.69 = 4.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0148989, upper bound: 0.0148989

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142101, upper bound: 0.0142089
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142089, upper bound: 0.0142101
time: 3.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.60 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.60
Output dim: 9, lower bound: -0.0142101, upper bound: 0.0142089
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.60
Output dim: 9, lower bound: -0.0142089, upper bound: 0.0142101

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070288, 0.0070771
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140544, upper bound: 0.0140972
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140986, upper bound: 0.0140534
time: 2.92 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070595, 0.0070288
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140534, upper bound: 0.0140985
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140972, upper bound: 0.0140544
time: 1.58 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.87 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 9, lower bound: -0.0140544, upper bound: 0.0140972
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 9, lower bound: -0.0140986, upper bound: 0.0140534
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 9, lower bound: -0.0140534, upper bound: 0.0140985
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 9, lower bound: -0.0140972, upper bound: 0.0140544

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070288, 0.0070770
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070287, 0.0070771
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070596, 0.0070287
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070595, 0.0070288
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440
time: 1.75 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129440, upper bound: 0.0129685
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.45
Output dim: 9, lower bound: -0.0129685, upper bound: 0.0129440

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.48 + 33.49 = 37.97 seconds
