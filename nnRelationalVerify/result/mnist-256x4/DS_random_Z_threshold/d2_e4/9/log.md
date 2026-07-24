## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.279429504


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142)
1: (-0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979)
2: (-0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389)
3: (-0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414)
4: (-0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357)
5: (-0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473)
6: (-0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181)
7: (-0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449)
8: (-0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810)
9: (0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.84 + 3.41 = 4.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.2910724, upper bound: 0.2910724

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2902099, upper bound: 0.2902099
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2902099, upper bound: 0.2902099
time: 1.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.82
Output dim: 9, lower bound: -0.2902099, upper bound: 0.2902099
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.82
Output dim: 9, lower bound: -0.2902099, upper bound: 0.2902099

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2878840, upper bound: 0.2880921
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2880912, upper bound: 0.2878809
time: 1.89 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2902099, upper bound: 0.2901292
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2901293, upper bound: 0.2902099
time: 2.32 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.56 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.56
Output dim: 9, lower bound: -0.2878840, upper bound: 0.2880921
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.56
Output dim: 9, lower bound: -0.2880912, upper bound: 0.2878809
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.56
Output dim: 9, lower bound: -0.2902099, upper bound: 0.2901292
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.56
Output dim: 9, lower bound: -0.2901293, upper bound: 0.2902099

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2761413, upper bound: 0.2762494
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2761413, upper bound: 0.2762494
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2880912, upper bound: 0.2877764
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2880273, upper bound: 0.2878809
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2890684, upper bound: 0.2890677
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2891350, upper bound: 0.2889870
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2787386, upper bound: 0.2787870
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2787386, upper bound: 0.2787870
time: 1.52 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.88 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2761413, upper bound: 0.2762494
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2761413, upper bound: 0.2762494
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2880912, upper bound: 0.2877764
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2880273, upper bound: 0.2878809
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2890684, upper bound: 0.2890677
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2891350, upper bound: 0.2889870
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2787386, upper bound: 0.2787870
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.88
Output dim: 9, lower bound: -0.2787386, upper bound: 0.2787870

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2866232, upper bound: 0.2862337
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2866198, upper bound: 0.2862593
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2878717, upper bound: 0.2877420
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2878873, upper bound: 0.2877181
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2865293, upper bound: 0.2866834
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2866905, upper bound: 0.2865372
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2891350, upper bound: 0.2887045
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2888615, upper bound: 0.2889870
time: 1.97 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.68 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2866232, upper bound: 0.2862337
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2866198, upper bound: 0.2862593
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2878717, upper bound: 0.2877420
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2878873, upper bound: 0.2877181
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2865293, upper bound: 0.2866834
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2866905, upper bound: 0.2865372
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2891350, upper bound: 0.2887045
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 9, lower bound: -0.2888615, upper bound: 0.2889870

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2669958, upper bound: 0.2669252
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2669958, upper bound: 0.2669252
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2858185, upper bound: 0.2855533
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2859130, upper bound: 0.2854647
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2876415, upper bound: 0.2875627
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2876199, upper bound: 0.2875905
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2878794, upper bound: 0.2876861
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2878483, upper bound: 0.2877139
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2780880, upper bound: 0.2782401
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2780880, upper bound: 0.2782401
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2782346, upper bound: 0.2781070
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2782346, upper bound: 0.2781070
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2865953, upper bound: 0.2863112
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2867547, upper bound: 0.2861952
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2802745, upper bound: 0.2803827
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2802745, upper bound: 0.2803827
time: 2.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.91 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2669958, upper bound: 0.2669252
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2669958, upper bound: 0.2669252
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2858185, upper bound: 0.2855533
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2859130, upper bound: 0.2854647
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2876415, upper bound: 0.2875627
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2876199, upper bound: 0.2875905
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2878794, upper bound: 0.2876861
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2878483, upper bound: 0.2877139
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2780880, upper bound: 0.2782401
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2780880, upper bound: 0.2782401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2782346, upper bound: 0.2781070
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2782346, upper bound: 0.2781070
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2865953, upper bound: 0.2863112
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2867547, upper bound: 0.2861952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2802745, upper bound: 0.2803827
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 9, lower bound: -0.2802745, upper bound: 0.2803827

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2811444, upper bound: 0.2809463
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2811466, upper bound: 0.2809301
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2740005, upper bound: 0.2737101
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2740005, upper bound: 0.2737101
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2862521, upper bound: 0.2861107
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2862558, upper bound: 0.2861473
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2870396, upper bound: 0.2870260
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2870483, upper bound: 0.2870260
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2784269, upper bound: 0.2784071
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2784269, upper bound: 0.2784071
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2789021, upper bound: 0.2788353
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2789021, upper bound: 0.2788353
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2850154, upper bound: 0.2847074
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2850115, upper bound: 0.2847073
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2665835, upper bound: 0.2663143
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2665835, upper bound: 0.2663143
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2532940, upper bound: 0.2532829
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2532940, upper bound: 0.2532829
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2801329, upper bound: 0.2802480
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2801390, upper bound: 0.2802309
time: 2.47 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.90 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2811444, upper bound: 0.2809463
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2811466, upper bound: 0.2809301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2740005, upper bound: 0.2737101
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2740005, upper bound: 0.2737101
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2862521, upper bound: 0.2861107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2862558, upper bound: 0.2861473
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2870396, upper bound: 0.2870260
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2870483, upper bound: 0.2870260
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2784269, upper bound: 0.2784071
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2784269, upper bound: 0.2784071
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2789021, upper bound: 0.2788353
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2789021, upper bound: 0.2788353
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2850154, upper bound: 0.2847074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2850115, upper bound: 0.2847073
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2665835, upper bound: 0.2663143
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2665835, upper bound: 0.2663143
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2532940, upper bound: 0.2532829
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2532940, upper bound: 0.2532829
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2801329, upper bound: 0.2802480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 9, lower bound: -0.2801390, upper bound: 0.2802309

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2636981, upper bound: 0.2636293
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2636981, upper bound: 0.2636293
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2668773, upper bound: 0.2667079
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2668773, upper bound: 0.2667079
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2836921, upper bound: 0.2836980
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2838528, upper bound: 0.2835664
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2837051, upper bound: 0.2837320
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2838612, upper bound: 0.2835923
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2663618, upper bound: 0.2663696
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2663618, upper bound: 0.2663696
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2856299, upper bound: 0.2855331
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2856323, upper bound: 0.2855564
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2819455, upper bound: 0.2817109
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2819617, upper bound: 0.2816684
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2696052, upper bound: 0.2694299
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2696052, upper bound: 0.2694299
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2801123, upper bound: 0.2802004
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2801116, upper bound: 0.2802263
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2599284, upper bound: 0.2599609
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2599284, upper bound: 0.2599609
time: 1.72 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.24 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2636981, upper bound: 0.2636293
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2636981, upper bound: 0.2636293
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2668773, upper bound: 0.2667079
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2668773, upper bound: 0.2667079
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2836921, upper bound: 0.2836980
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2838528, upper bound: 0.2835664
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2837051, upper bound: 0.2837320
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2838612, upper bound: 0.2835923
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2663618, upper bound: 0.2663696
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2663618, upper bound: 0.2663696
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2856299, upper bound: 0.2855331
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2856323, upper bound: 0.2855564
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2819455, upper bound: 0.2817109
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2819617, upper bound: 0.2816684
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2696052, upper bound: 0.2694299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2696052, upper bound: 0.2694299
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2801123, upper bound: 0.2802004
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2801116, upper bound: 0.2802263
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2599284, upper bound: 0.2599609
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.24
Output dim: 9, lower bound: -0.2599284, upper bound: 0.2599609

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2830417, upper bound: 0.2830693
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2830642, upper bound: 0.2830584
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2838528, upper bound: 0.2833773
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2836219, upper bound: 0.2835663
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2830418, upper bound: 0.2830928
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2830701, upper bound: 0.2830898
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2644002, upper bound: 0.2641832
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2644002, upper bound: 0.2641832
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2848580, upper bound: 0.2848487
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2849341, upper bound: 0.2847596
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2844405, upper bound: 0.2844238
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2845437, upper bound: 0.2843687
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2695487, upper bound: 0.2694181
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2695487, upper bound: 0.2694181
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2813346, upper bound: 0.2810491
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2813434, upper bound: 0.2810117
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2776964, upper bound: 0.2778789
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2777917, upper bound: 0.2777438
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2674077, upper bound: 0.2674920
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2674077, upper bound: 0.2674921
time: 1.74 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.12 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2830417, upper bound: 0.2830693
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2830642, upper bound: 0.2830584
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2838528, upper bound: 0.2833773
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2836219, upper bound: 0.2835663
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2830418, upper bound: 0.2830928
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2830701, upper bound: 0.2830898
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2644002, upper bound: 0.2641832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2644002, upper bound: 0.2641832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2848580, upper bound: 0.2848487
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2849341, upper bound: 0.2847596
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2844405, upper bound: 0.2844238
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2845437, upper bound: 0.2843687
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2695487, upper bound: 0.2694181
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2695487, upper bound: 0.2694181
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2813346, upper bound: 0.2810491
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2813434, upper bound: 0.2810117
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2776964, upper bound: 0.2778789
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2777917, upper bound: 0.2777438
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2674077, upper bound: 0.2674920
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 9, lower bound: -0.2674077, upper bound: 0.2674921

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2818405, upper bound: 0.2819463
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2819415, upper bound: 0.2818836
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2802410, upper bound: 0.2802205
time: 5.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2802565, upper bound: 0.2802219
time: 4.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2633154, upper bound: 0.2630318
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2633154, upper bound: 0.2630318
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2790365, upper bound: 0.2789987
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2790365, upper bound: 0.2789987
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2700903, upper bound: 0.2701886
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2700903, upper bound: 0.2701886
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2714724, upper bound: 0.2714268
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2714724, upper bound: 0.2714268
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2803107, upper bound: 0.2802742
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2803107, upper bound: 0.2802742
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2732038, upper bound: 0.2731022
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2732038, upper bound: 0.2731022
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2649477, upper bound: 0.2648282
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2649477, upper bound: 0.2648282
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2800808, upper bound: 0.2798574
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2800808, upper bound: 0.2798574
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2728052, upper bound: 0.2726446
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2728052, upper bound: 0.2726446
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2702662, upper bound: 0.2700218
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2702662, upper bound: 0.2700218
time: 1.65 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.09 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2818405, upper bound: 0.2819463
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2819415, upper bound: 0.2818836
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2802410, upper bound: 0.2802205
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2802565, upper bound: 0.2802219
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2633154, upper bound: 0.2630318
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2633154, upper bound: 0.2630318
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2790365, upper bound: 0.2789987
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2790365, upper bound: 0.2789987
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2700903, upper bound: 0.2701886
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2700903, upper bound: 0.2701886
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2714724, upper bound: 0.2714268
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2714724, upper bound: 0.2714268
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2803107, upper bound: 0.2802742
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2803107, upper bound: 0.2802742
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2732038, upper bound: 0.2731022
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2732038, upper bound: 0.2731022
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2649477, upper bound: 0.2648282
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2649477, upper bound: 0.2648282
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2800808, upper bound: 0.2798574
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2800808, upper bound: 0.2798574
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2728052, upper bound: 0.2726446
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2728052, upper bound: 0.2726446
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2702662, upper bound: 0.2700218
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.09
Output dim: 9, lower bound: -0.2702662, upper bound: 0.2700218

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2654544, upper bound: 0.2654726
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2654544, upper bound: 0.2654726
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2746559, upper bound: 0.2746472
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2746559, upper bound: 0.2746472
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2675791, upper bound: 0.2675525
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2675791, upper bound: 0.2675525
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2697518, upper bound: 0.2698348
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2697518, upper bound: 0.2698348
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2717568, upper bound: 0.2718745
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2717568, upper bound: 0.2718745
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2615168, upper bound: 0.2614822
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2615168, upper bound: 0.2614822
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142
1: -0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979
2: -0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389
3: -0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414
4: -0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357
5: -0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473
6: -0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181
7: -0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449
8: -0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810
9: 0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142
time: 1.85 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 4.52 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2654544, upper bound: 0.2654726
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2654544, upper bound: 0.2654726
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2746559, upper bound: 0.2746472
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2746559, upper bound: 0.2746472
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2675791, upper bound: 0.2675525
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2675791, upper bound: 0.2675525
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2697518, upper bound: 0.2698348
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2697518, upper bound: 0.2698348
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2717568, upper bound: 0.2718745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2717568, upper bound: 0.2718745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2615168, upper bound: 0.2614822
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2615168, upper bound: 0.2614822
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.52
Output dim: 9, lower bound: -0.2773318, upper bound: 0.2770142

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.25 + 325.40 = 329.64 seconds
