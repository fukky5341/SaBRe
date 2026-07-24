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
execution time: IAR + RelationalAnalysis = 0.95 + 2.47 = 3.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0148989, upper bound: 0.0148989

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146917, upper bound: 0.0146917
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146917, upper bound: 0.0146917
time: 2.02 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.31
Output dim: 9, lower bound: -0.0146917, upper bound: 0.0146917
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.31
Output dim: 9, lower bound: -0.0146917, upper bound: 0.0146917

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070285, 0.0070317
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142662, upper bound: 0.0143899
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143900, upper bound: 0.0142662
time: 2.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070317, 0.0070285
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0131775
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0131775
time: 1.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.98 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 9, lower bound: -0.0142662, upper bound: 0.0143899
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 9, lower bound: -0.0143900, upper bound: 0.0142662
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.98
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0131775
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.98
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0131775

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0069900, 0.0069903
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135304, upper bound: 0.0136404
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134747, upper bound: 0.0136796
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0069870, 0.0069923
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130781, upper bound: 0.0130493
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130781, upper bound: 0.0130493
time: 1.98 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.84 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.84
Output dim: 9, lower bound: -0.0135304, upper bound: 0.0136404
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.84
Output dim: 9, lower bound: -0.0134747, upper bound: 0.0136796
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.84
Output dim: 9, lower bound: -0.0130781, upper bound: 0.0130493
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.84
Output dim: 9, lower bound: -0.0130781, upper bound: 0.0130493

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0069793, 0.0069883
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129576, upper bound: 0.0131426
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130102, upper bound: 0.0130704
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916
1: -0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314
2: 0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552
3: -0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636
4: -0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996
5: -0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761
6: -0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0069900, 0.0069796
7: -0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231
8: -0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160
9: 0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0118693, upper bound: 0.0119084
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0118693, upper bound: 0.0119084
time: 1.58 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.06
Output dim: 9, lower bound: -0.0129576, upper bound: 0.0131426
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.06
Output dim: 9, lower bound: -0.0130102, upper bound: 0.0130704
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.06
Output dim: 9, lower bound: -0.0118693, upper bound: 0.0119084
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.06
Output dim: 9, lower bound: -0.0118693, upper bound: 0.0119084

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.42 + 32.08 = 35.50 seconds
