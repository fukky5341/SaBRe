## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000711535


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0031298, 0.0031298)
1: (-0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008824, 0.0008824)
2: (0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0065105, 0.0065105)
3: (0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008616, 0.0008616)
4: (-0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0048656, 0.0048656)
5: (0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013518, 0.0013518)
6: (0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012270, 0.0012270)
7: (-0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045790, 0.0045790)
8: (-0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035639, 0.0035639)
9: (-0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003075, 0.0003075)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 2.92 = 4.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0008371, upper bound: 0.0008363

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008023, upper bound: 0.0007908
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007903, upper bound: 0.0008019
time: 2.00 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.81 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.81
Output dim: 5, lower bound: -0.0008023, upper bound: 0.0007908
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.81
Output dim: 5, lower bound: -0.0007903, upper bound: 0.0008019

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030920, 0.0031167
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008717, 0.0008787
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0064320, 0.0064834
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008512, 0.0008580
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0048453, 0.0048068
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013462, 0.0013355
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012219, 0.0012122
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045600, 0.0045238
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035209, 0.0035490
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003062, 0.0003038

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007940, upper bound: 0.0007791
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007897, upper bound: 0.0007817
time: 1.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0031298, 0.0030920
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008824, 0.0008717
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0065105, 0.0064320
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008616, 0.0008512
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0048068, 0.0048656
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013355, 0.0013518
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012122, 0.0012270
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045238, 0.0045790
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035639, 0.0035209
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003038, 0.0003075

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007816, upper bound: 0.0007897
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007792, upper bound: 0.0007944
time: 1.85 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 5, lower bound: -0.0007940, upper bound: 0.0007791
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 5, lower bound: -0.0007897, upper bound: 0.0007817
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 5, lower bound: -0.0007816, upper bound: 0.0007897
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 5, lower bound: -0.0007792, upper bound: 0.0007944

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030623, 0.0030966
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008634, 0.0008730
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063702, 0.0064415
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008430, 0.0008524
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0048140, 0.0047607
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013375, 0.0013227
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012140, 0.0012006
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045305, 0.0044804
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034871, 0.0035261
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003042, 0.0003008

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030707, 0.0030870
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008657, 0.0008704
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063876, 0.0064217
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008453, 0.0008498
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047992, 0.0047737
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013334, 0.0013263
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012103, 0.0012039
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045166, 0.0044926
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034966, 0.0035152
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003033, 0.0003017

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030996, 0.0030707
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008739, 0.0008657
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0064477, 0.0063876
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008533, 0.0008453
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047737, 0.0048186
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013263, 0.0013388
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012039, 0.0012152
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044926, 0.0045349
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035295, 0.0034966
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003017, 0.0003045

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0031079, 0.0030623
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008762, 0.0008634
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0064651, 0.0063702
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008556, 0.0008430
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047607, 0.0048316
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013227, 0.0013424
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012006, 0.0012185
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044804, 0.0045471
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035390, 0.0034871
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003008, 0.0003053

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859
time: 1.61 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.52 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006853
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006859

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.24 + 31.44 = 35.68 seconds
