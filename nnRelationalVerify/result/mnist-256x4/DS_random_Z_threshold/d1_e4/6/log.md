## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.017292288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797)
1: (0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944)
2: (-0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0199057, 0.0199058)
3: (-0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347)
4: (-0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642)
5: (-0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152)
6: (-0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641)
7: (-0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542)
8: (-0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295680, 0.0295680)
9: (-0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.96 + 3.36 = 4.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0180128, upper bound: 0.0180128

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179335, upper bound: 0.0179863
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179863, upper bound: 0.0179335
time: 2.70 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.73 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.73
Output dim: 1, lower bound: -0.0179335, upper bound: 0.0179863
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.73
Output dim: 1, lower bound: -0.0179863, upper bound: 0.0179335

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198984, 0.0199017
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295673, 0.0295665
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179129, upper bound: 0.0179783
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179255, upper bound: 0.0179584
time: 2.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0199017, 0.0198984
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295665, 0.0295673
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150053, upper bound: 0.0150002
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150053, upper bound: 0.0150002
time: 1.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.18 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 1, lower bound: -0.0179129, upper bound: 0.0179783
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 1, lower bound: -0.0179255, upper bound: 0.0179584
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.18
Output dim: 1, lower bound: -0.0150053, upper bound: 0.0150002
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.18
Output dim: 1, lower bound: -0.0150053, upper bound: 0.0150002

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198987, 0.0199026
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295678, 0.0295668
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178780, upper bound: 0.0179696
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178780, upper bound: 0.0179276
time: 3.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198993, 0.0199019
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295677, 0.0295670
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178090, upper bound: 0.0178650
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178304, upper bound: 0.0178401
time: 3.22 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 7.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.22
Output dim: 1, lower bound: -0.0178780, upper bound: 0.0179696
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.22
Output dim: 1, lower bound: -0.0178780, upper bound: 0.0179276
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.22
Output dim: 1, lower bound: -0.0178090, upper bound: 0.0178650
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.22
Output dim: 1, lower bound: -0.0178304, upper bound: 0.0178401

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198932, 0.0198983
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295665, 0.0295652
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175360, upper bound: 0.0176147
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175399, upper bound: 0.0176053
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198942, 0.0198972
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295662, 0.0295654
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178914, upper bound: 0.0179076
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178898, upper bound: 0.0179156
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198687, 0.0198775
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295607, 0.0295583
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169884, upper bound: 0.0170334
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169884, upper bound: 0.0170336
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198745, 0.0198714
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295590, 0.0295599
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173453, upper bound: 0.0173624
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173453, upper bound: 0.0173624
time: 2.54 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0175360, upper bound: 0.0176147
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0175399, upper bound: 0.0176053
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0178914, upper bound: 0.0179076
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0178898, upper bound: 0.0179156
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0169884, upper bound: 0.0170334
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0169884, upper bound: 0.0170336
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0173453, upper bound: 0.0173624
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.13
Output dim: 1, lower bound: -0.0173453, upper bound: 0.0173624

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198918, 0.0198973
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295662, 0.0295647
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174620, upper bound: 0.0175359
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174578, upper bound: 0.0175379
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198923, 0.0198968
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295661, 0.0295649
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175281, upper bound: 0.0175864
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175174, upper bound: 0.0175938
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198763, 0.0198802
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295643, 0.0295633
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171299, upper bound: 0.0171349
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171299, upper bound: 0.0171348
time: 2.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198773, 0.0198794
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295641, 0.0295635
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178323, upper bound: 0.0178881
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178601, upper bound: 0.0178893
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198622, 0.0198583
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295556, 0.0295567
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173033, upper bound: 0.0173513
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173342, upper bound: 0.0173161
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198615, 0.0198714
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295590, 0.0295565
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0166399, upper bound: 0.0166418
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0166399, upper bound: 0.0166418
time: 2.31 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0174620, upper bound: 0.0175359
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0174578, upper bound: 0.0175379
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0175281, upper bound: 0.0175864
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0175174, upper bound: 0.0175938
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0171299, upper bound: 0.0171349
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0171299, upper bound: 0.0171348
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0178323, upper bound: 0.0178881
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0178601, upper bound: 0.0178893
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0173033, upper bound: 0.0173513
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0173342, upper bound: 0.0173161
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0166399, upper bound: 0.0166418
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.51
Output dim: 1, lower bound: -0.0166399, upper bound: 0.0166418

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198465, 0.0198532
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295573, 0.0295555
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173458, upper bound: 0.0174544
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173431, upper bound: 0.0174204
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198476, 0.0198522
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295570, 0.0295558
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168599, upper bound: 0.0169400
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168599, upper bound: 0.0169400
time: 2.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198743, 0.0198798
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295642, 0.0295627
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175167, upper bound: 0.0175743
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175170, upper bound: 0.0175741
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198753, 0.0198790
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295640, 0.0295630
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173079, upper bound: 0.0173795
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173051, upper bound: 0.0173860
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198680, 0.0198718
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295612, 0.0295602
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175544, upper bound: 0.0175793
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175544, upper bound: 0.0175793
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198698, 0.0198701
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295608, 0.0295607
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176834, upper bound: 0.0178643
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0178354, upper bound: 0.0177082
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198213, 0.0198358
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295418, 0.0295380
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172771, upper bound: 0.0173248
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172772, upper bound: 0.0173236
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198377, 0.0198174
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295369, 0.0295423
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173005, upper bound: 0.0173138
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173319, upper bound: 0.0172904
time: 2.58 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0173458, upper bound: 0.0174544
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0173431, upper bound: 0.0174204
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0168599, upper bound: 0.0169400
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0168599, upper bound: 0.0169400
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0175167, upper bound: 0.0175743
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0175170, upper bound: 0.0175741
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0173079, upper bound: 0.0173795
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0173051, upper bound: 0.0173860
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0175544, upper bound: 0.0175793
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0175544, upper bound: 0.0175793
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0176834, upper bound: 0.0178643
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0178354, upper bound: 0.0177082
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0172771, upper bound: 0.0173248
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0172772, upper bound: 0.0173236
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0173005, upper bound: 0.0173138
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.59
Output dim: 1, lower bound: -0.0173319, upper bound: 0.0172904

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198280, 0.0198363
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295526, 0.0295504
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169976, upper bound: 0.0173343
time: 5.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172177, upper bound: 0.0171036
time: 3.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198298, 0.0198347
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295522, 0.0295509
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167470, upper bound: 0.0168075
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167470, upper bound: 0.0168075
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198693, 0.0198754
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295624, 0.0295607
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173187, upper bound: 0.0173767
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173233, upper bound: 0.0173735
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198702, 0.0198747
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295622, 0.0295610
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174413, upper bound: 0.0174946
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174377, upper bound: 0.0174956
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198559, 0.0198605
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295591, 0.0295578
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 136

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171540, upper bound: 0.0172280
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171631, upper bound: 0.0172189
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198568, 0.0198607
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295591, 0.0295581
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170520, upper bound: 0.0171267
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170520, upper bound: 0.0171265
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198472, 0.0198473
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295549, 0.0295549
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174238, upper bound: 0.0174819
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174524, upper bound: 0.0174453
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198435, 0.0198499
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295556, 0.0295539
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175024, upper bound: 0.0175683
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175435, upper bound: 0.0175264
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198473, 0.0198550
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295542, 0.0295521
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176162, upper bound: 0.0178516
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176162, upper bound: 0.0178514
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198552, 0.0198476
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295522, 0.0295542
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173213, upper bound: 0.0173982
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173213, upper bound: 0.0173982
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198049, 0.0198196
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295395, 0.0295356
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172545, upper bound: 0.0173248
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172771, upper bound: 0.0172984
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198051, 0.0198210
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295399, 0.0295356
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171413, upper bound: 0.0172259
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171764, upper bound: 0.0171812
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198176, 0.0198033
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295305, 0.0295344
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172886, upper bound: 0.0172959
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172858, upper bound: 0.0173021
time: 2.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198235, 0.0197974
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295290, 0.0295359
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0164060, upper bound: 0.0163923
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0164060, upper bound: 0.0163923
time: 2.46 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.79 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0169976, upper bound: 0.0173343
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0172177, upper bound: 0.0171036
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0167470, upper bound: 0.0168075
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0167470, upper bound: 0.0168075
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0173187, upper bound: 0.0173767
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0173233, upper bound: 0.0173735
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0174413, upper bound: 0.0174946
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0174377, upper bound: 0.0174956
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0171540, upper bound: 0.0172280
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0171631, upper bound: 0.0172189
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0170520, upper bound: 0.0171267
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0170520, upper bound: 0.0171265
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0174238, upper bound: 0.0174819
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0174524, upper bound: 0.0174453
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0175024, upper bound: 0.0175683
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0175435, upper bound: 0.0175264
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0176162, upper bound: 0.0178516
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0176162, upper bound: 0.0178514
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0173213, upper bound: 0.0173982
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0173213, upper bound: 0.0173982
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0172545, upper bound: 0.0173248
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0172771, upper bound: 0.0172984
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0171413, upper bound: 0.0172259
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0171764, upper bound: 0.0171812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0172886, upper bound: 0.0172959
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0172858, upper bound: 0.0173021
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0164060, upper bound: 0.0163923
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.79
Output dim: 1, lower bound: -0.0164060, upper bound: 0.0163923

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0196612, 0.0197157
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295103, 0.0294957
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167939, upper bound: 0.0171161
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167894, upper bound: 0.0171188
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198683, 0.0198747
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295622, 0.0295605
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170532, upper bound: 0.0171046
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170506, upper bound: 0.0171055
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198685, 0.0198743
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295621, 0.0295605
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171893, upper bound: 0.0172622
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172113, upper bound: 0.0172460
time: 2.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198270, 0.0198322
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295533, 0.0295519
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173432, upper bound: 0.0173899
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173433, upper bound: 0.0173898
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198277, 0.0198313
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295530, 0.0295520
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173397, upper bound: 0.0173905
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173398, upper bound: 0.0173903
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197807, 0.0198033
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295391, 0.0295331
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174089, upper bound: 0.0174692
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174114, upper bound: 0.0174692
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197987, 0.0197809
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295331, 0.0295379
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172624, upper bound: 0.0173057
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172629, upper bound: 0.0173050
time: 3.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198020, 0.0198245
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295412, 0.0295352
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0166750, upper bound: 0.0167312
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0166750, upper bound: 0.0167312
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198197, 0.0198084
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295369, 0.0295399
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174077, upper bound: 0.0173992
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174079, upper bound: 0.0173992
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198426, 0.0198512
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295525, 0.0295502
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0149066, upper bound: 0.0149128
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0149066, upper bound: 0.0149128
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198435, 0.0198503
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295522, 0.0295504
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173968, upper bound: 0.0176258
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173968, upper bound: 0.0176254
time: 2.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198351, 0.0198233
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295458, 0.0295489
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173843, upper bound: 0.0172492
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173709, upper bound: 0.0172502
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198309, 0.0198265
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295467, 0.0295478
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175044, upper bound: 0.0173957
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172838, upper bound: 0.0173630
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197985, 0.0198133
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295375, 0.0295335
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171485, upper bound: 0.0172334
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171560, upper bound: 0.0172124
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197987, 0.0198116
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295370, 0.0295336
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 59

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170068, upper bound: 0.0172026
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171866, upper bound: 0.0170348
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197958, 0.0197830
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295286, 0.0295320
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170373, upper bound: 0.0170417
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170373, upper bound: 0.0170417
time: 3.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197974, 0.0197828
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295285, 0.0295324
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170867, upper bound: 0.0172780
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172621, upper bound: 0.0171261
time: 2.83 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 7.86 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0167939, upper bound: 0.0171161
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0167894, upper bound: 0.0171188
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0170532, upper bound: 0.0171046
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0170506, upper bound: 0.0171055
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0171893, upper bound: 0.0172622
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0172113, upper bound: 0.0172460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173432, upper bound: 0.0173899
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173433, upper bound: 0.0173898
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173397, upper bound: 0.0173905
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173398, upper bound: 0.0173903
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0174089, upper bound: 0.0174692
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0174114, upper bound: 0.0174692
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0172624, upper bound: 0.0173057
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0172629, upper bound: 0.0173050
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0166750, upper bound: 0.0167312
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0166750, upper bound: 0.0167312
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0174077, upper bound: 0.0173992
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0174079, upper bound: 0.0173992
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0149066, upper bound: 0.0149128
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0149066, upper bound: 0.0149128
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173968, upper bound: 0.0176258
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173968, upper bound: 0.0176254
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173843, upper bound: 0.0172492
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0173709, upper bound: 0.0172502
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0175044, upper bound: 0.0173957
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0172838, upper bound: 0.0173630
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0171485, upper bound: 0.0172334
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0171560, upper bound: 0.0172124
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0170068, upper bound: 0.0172026
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0171866, upper bound: 0.0170348
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0170373, upper bound: 0.0170417
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0170373, upper bound: 0.0170417
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0170867, upper bound: 0.0172780
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.86
Output dim: 1, lower bound: -0.0172621, upper bound: 0.0171261

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198251, 0.0198307
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295529, 0.0295514
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171560, upper bound: 0.0172044
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171595, upper bound: 0.0172019
time: 4.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198254, 0.0198304
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295528, 0.0295515
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161902, upper bound: 0.0162284
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161902, upper bound: 0.0162284
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198259, 0.0198298
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295526, 0.0295516
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171440, upper bound: 0.0171721
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171060, upper bound: 0.0171934
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198262, 0.0198294
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295525, 0.0295516
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0162383, upper bound: 0.0162886
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0162383, upper bound: 0.0162886
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197755, 0.0197985
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295373, 0.0295311
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172594, upper bound: 0.0173799
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173143, upper bound: 0.0173542
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197767, 0.0197980
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295372, 0.0295315
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0165760, upper bound: 0.0166284
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0165760, upper bound: 0.0166284
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197969, 0.0197794
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295327, 0.0295374
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172242, upper bound: 0.0172267
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172383, upper bound: 0.0172054
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197972, 0.0197790
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295326, 0.0295375
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171374, upper bound: 0.0171038
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170611, upper bound: 0.0171242
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198177, 0.0198069
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295365, 0.0295394
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172284, upper bound: 0.0171931
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171994, upper bound: 0.0172185
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198181, 0.0198065
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295364, 0.0295395
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172516, upper bound: 0.0172294
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172509, upper bound: 0.0172382
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198207, 0.0198293
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295466, 0.0295444
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172541, upper bound: 0.0174770
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172971, upper bound: 0.0174767
time: 3.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198226, 0.0198280
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295463, 0.0295448
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173625, upper bound: 0.0176230
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173625, upper bound: 0.0176180
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198098, 0.0197863
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295361, 0.0295423
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168992, upper bound: 0.0167794
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168992, upper bound: 0.0167794
time: 2.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197980, 0.0197904
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295372, 0.0295392
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171668, upper bound: 0.0170410
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171677, upper bound: 0.0170395
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198114, 0.0198135
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295406, 0.0295400
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174039, upper bound: 0.0173017
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174066, upper bound: 0.0172951
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198155, 0.0198070
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295388, 0.0295411
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174708, upper bound: 0.0173516
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175140, upper bound: 0.0173220
time: 2.69 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 5.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171560, upper bound: 0.0172044
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171595, upper bound: 0.0172019
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0161902, upper bound: 0.0162284
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0161902, upper bound: 0.0162284
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171440, upper bound: 0.0171721
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171060, upper bound: 0.0171934
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0162383, upper bound: 0.0162886
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0162383, upper bound: 0.0162886
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172594, upper bound: 0.0173799
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0173143, upper bound: 0.0173542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0165760, upper bound: 0.0166284
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0165760, upper bound: 0.0166284
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172242, upper bound: 0.0172267
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172383, upper bound: 0.0172054
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171374, upper bound: 0.0171038
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0170611, upper bound: 0.0171242
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172284, upper bound: 0.0171931
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171994, upper bound: 0.0172185
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172516, upper bound: 0.0172294
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172509, upper bound: 0.0172382
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172541, upper bound: 0.0174770
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0172971, upper bound: 0.0174767
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0173625, upper bound: 0.0176230
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0173625, upper bound: 0.0176180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0168992, upper bound: 0.0167794
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0168992, upper bound: 0.0167794
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171668, upper bound: 0.0170410
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0171677, upper bound: 0.0170395
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0174039, upper bound: 0.0173017
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0174066, upper bound: 0.0172951
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0174708, upper bound: 0.0173516
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.74
Output dim: 1, lower bound: -0.0175140, upper bound: 0.0173220

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197646, 0.0197887
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295347, 0.0295283
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170871, upper bound: 0.0171889
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170871, upper bound: 0.0171873
time: 2.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197653, 0.0197876
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295344, 0.0295285
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171198, upper bound: 0.0171384
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171197, upper bound: 0.0171436
time: 3.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198188, 0.0198278
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295462, 0.0295438
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0171449, upper bound: 0.0173986
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172150, upper bound: 0.0173715
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198192, 0.0198274
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295461, 0.0295440
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170153, upper bound: 0.0172381
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170153, upper bound: 0.0172626
time: 2.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198031, 0.0198124
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295395, 0.0295370
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169627, upper bound: 0.0175057
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172856, upper bound: 0.0172574
time: 2.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0198089, 0.0198086
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295385, 0.0295385
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0165562, upper bound: 0.0166972
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0165562, upper bound: 0.0166972
time: 2.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197847, 0.0197905
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295345, 0.0295330
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172694, upper bound: 0.0172057
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173067, upper bound: 0.0171663
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197884, 0.0197864
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295334, 0.0295339
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173938, upper bound: 0.0172779
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173948, upper bound: 0.0172810
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197726, 0.0197792
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295239, 0.0295222
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173684, upper bound: 0.0172593
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173721, upper bound: 0.0172571
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197906, 0.0197642
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295199, 0.0295269
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 59

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172364, upper bound: 0.0172308
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174345, upper bound: 0.0170941
time: 3.04 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 6.40 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0170871, upper bound: 0.0171889
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0170871, upper bound: 0.0171873
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0171198, upper bound: 0.0171384
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0171197, upper bound: 0.0171436
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0171449, upper bound: 0.0173986
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0172150, upper bound: 0.0173715
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0170153, upper bound: 0.0172381
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0170153, upper bound: 0.0172626
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0169627, upper bound: 0.0175057
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0172856, upper bound: 0.0172574
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0165562, upper bound: 0.0166972
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0165562, upper bound: 0.0166972
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0172694, upper bound: 0.0172057
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0173067, upper bound: 0.0171663
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0173938, upper bound: 0.0172779
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0173948, upper bound: 0.0172810
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0173684, upper bound: 0.0172593
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0173721, upper bound: 0.0172571
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0172364, upper bound: 0.0172308
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.40
Output dim: 1, lower bound: -0.0174345, upper bound: 0.0170941

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197862, 0.0198005
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295390, 0.0295352
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169531, upper bound: 0.0171701
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169551, upper bound: 0.0171657
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197920, 0.0197952
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295376, 0.0295368
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160687, upper bound: 0.0162055
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160687, upper bound: 0.0162057
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0196366, 0.0196900
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0294963, 0.0294820
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0164630, upper bound: 0.0169773
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0164630, upper bound: 0.0169773
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197358, 0.0197220
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295126, 0.0295163
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170809, upper bound: 0.0169179
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170638, upper bound: 0.0169417
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197832, 0.0197821
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295317, 0.0295320
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172287, upper bound: 0.0171048
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170110, upper bound: 0.0171022
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197842, 0.0197812
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295315, 0.0295323
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171319, upper bound: 0.0172522
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173683, upper bound: 0.0172168
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197465, 0.0197561
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295177, 0.0295151
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170086, upper bound: 0.0169164
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170115, upper bound: 0.0169158
time: 2.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197495, 0.0197520
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295166, 0.0295160
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172742, upper bound: 0.0171805
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172994, upper bound: 0.0171507
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0196903, 0.0196400
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0294820, 0.0294954
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169966, upper bound: 0.0170700
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174161, upper bound: 0.0170615
time: 2.96 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 6.69 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0169531, upper bound: 0.0171701
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0169551, upper bound: 0.0171657
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0160687, upper bound: 0.0162055
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0160687, upper bound: 0.0162057
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0164630, upper bound: 0.0169773
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0164630, upper bound: 0.0169773
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0170809, upper bound: 0.0169179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0170638, upper bound: 0.0169417
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0172287, upper bound: 0.0171048
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0170110, upper bound: 0.0171022
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0171319, upper bound: 0.0172522
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0173683, upper bound: 0.0172168
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0170086, upper bound: 0.0169164
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0170115, upper bound: 0.0169158
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0172742, upper bound: 0.0171805
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0172994, upper bound: 0.0171507
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0169966, upper bound: 0.0170700
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 6.69
Output dim: 1, lower bound: -0.0174161, upper bound: 0.0170615

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197787, 0.0197723
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295286, 0.0295303
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170875, upper bound: 0.0172052
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170875, upper bound: 0.0171814
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0197406, 0.0197416
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295138, 0.0295135
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170953, upper bound: 0.0169356
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170950, upper bound: 0.0169438
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797
1: 0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944
2: -0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0196813, 0.0196288
3: -0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347
4: -0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642
5: -0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152
6: -0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641
7: -0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542
8: -0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0294794, 0.0294934
9: -0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172210, upper bound: 0.0168349
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171961, upper bound: 0.0168663
time: 2.70 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 6.09 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 6.09
Output dim: 1, lower bound: -0.0170875, upper bound: 0.0172052
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 6.09
Output dim: 1, lower bound: -0.0170875, upper bound: 0.0171814
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 6.09
Output dim: 1, lower bound: -0.0170953, upper bound: 0.0169356
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 6.09
Output dim: 1, lower bound: -0.0170950, upper bound: 0.0169438
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 6.09
Output dim: 1, lower bound: -0.0172210, upper bound: 0.0168349
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 6.09
Output dim: 1, lower bound: -0.0171961, upper bound: 0.0168663

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.32 + 594.73 = 599.05 seconds
