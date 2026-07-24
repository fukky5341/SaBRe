## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.36165926200000004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9440718, 0.9440713)
1: (-13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1664696, 1.1664701)
2: (7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9627562, 0.9627564)
3: (-4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3745418, 1.3745418)
4: (-10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1760998, 1.1760998)
5: (-10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8981843, 0.8981843)
6: (-12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.6040778, 1.6040788)
7: (-3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1353478, 1.1353478)
8: (-1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0557561, 1.0557559)
9: (-8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9958510, 0.9958508)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.87 + 35.15 = 58.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3728446, upper bound: 0.3728441

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 470

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725382, upper bound: 0.3728440
time: 4.80 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728443, upper bound: 0.3725378
time: 6.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.61 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.61
Output dim: 2, lower bound: -0.3725382, upper bound: 0.3728440
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.61
Output dim: 2, lower bound: -0.3728443, upper bound: 0.3725378

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9385076, 0.9399402
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1584501, 1.1605172
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9616570, 0.9619420
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3739600, 1.3737602
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1673641, 1.1696196
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8968577, 0.8963964
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.6028142, 1.6031299
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1331382, 1.1323676
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0551453, 1.0549371
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9938765, 0.9931872

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3728423
time: 5.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
time: 3.32 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9399405, 0.9385076
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1605167, 1.1584506
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9619417, 0.9616573
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3737602, 1.3739595
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1696196, 1.1673641
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8963966, 0.8968577
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.6031299, 1.6028147
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1323676, 1.1331382
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0549374, 1.0551457
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9931870, 0.9938762

Time for backsubstitution: 21.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720429, upper bound: 0.3725377
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717366
time: 4.41 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.28 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.28
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3728423
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.28
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.28
Output dim: 2, lower bound: -0.3720429, upper bound: 0.3725377
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.28
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717366

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9352770, 0.9378781
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1582904, 1.1596899
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9580858, 0.9591827
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3760104, 1.3785038
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1563187, 1.1538887
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8955770, 0.8952742
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5540628, 1.5474286
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1195731, 1.1137433
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9991055, 1.0063117
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9972506, 0.9957652

Time for backsubstitution: 21.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3716998, upper bound: 0.3728083
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3716991, upper bound: 0.3725093
time: 4.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9364452, 0.9367096
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1576238, 1.1603575
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9588983, 0.9583704
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3787036, 1.3758106
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1516337, 1.1585736
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8957357, 0.8951154
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5471134, 1.5543785
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1145139, 1.1188025
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0065203, 0.9988973
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9964542, 0.9965613

Time for backsubstitution: 21.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3724998, upper bound: 0.3720099
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3724990, upper bound: 0.3717083
time: 5.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9367094, 0.9364455
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1603570, 1.1576242
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9583704, 0.9588983
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3758106, 1.3787036
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1585741, 1.1516337
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8951154, 0.8957355
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5543785, 1.5471134
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1188025, 1.1145139
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9988976, 1.0065203
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9965611, 0.9964545

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717082, upper bound: 0.3724986
time: 5.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720088, upper bound: 0.3725008
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9378786, 0.9352767
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1596904, 1.1582909
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9591830, 0.9580858
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3785038, 1.3760104
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1538887, 1.1563187
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8952746, 0.8955767
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5474291, 1.5540628
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1137433, 1.1195731
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0063114, 0.9991059
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9957657, 0.9972506

Time for backsubstitution: 21.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725081, upper bound: 0.3716989
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728087, upper bound: 0.3716996
time: 6.17 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.97 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3716998, upper bound: 0.3728083
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3716991, upper bound: 0.3725093
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3724998, upper bound: 0.3720099
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3724990, upper bound: 0.3717083
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3717082, upper bound: 0.3724986
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3720088, upper bound: 0.3725008
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3725081, upper bound: 0.3716989
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.97
Output dim: 2, lower bound: -0.3728087, upper bound: 0.3716996

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9352760, 0.9378769
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1582866, 1.1596880
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9580777, 0.9591794
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3760080, 1.3785028
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1563163, 1.1538877
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8955703, 0.8952715
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5540552, 1.5474253
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1195707, 1.1137428
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9991016, 1.0063100
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9972467, 0.9957554

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.53 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3562395, upper bound: 0.3617452
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3606381, upper bound: 0.3573459
time: 4.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9352756, 0.9378765
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1582837, 1.1596861
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9580824, 0.9591749
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3760071, 1.3785014
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1563144, 1.1538863
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8955665, 0.8952677
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5540504, 1.5474210
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1195693, 1.1137414
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9990993, 1.0063076
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9972410, 0.9957612

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.42 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3562387, upper bound: 0.3614429
time: 7.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3606373, upper bound: 0.3570436
time: 3.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9364448, 0.9367085
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1576200, 1.1603556
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9588902, 0.9583669
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3787012, 1.3758097
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1516309, 1.1585727
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8957295, 0.8951128
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5471058, 1.5543747
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1145120, 1.1188016
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0065160, 0.9988956
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9964504, 0.9965515

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.48 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3570359, upper bound: 0.3609468
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3614352, upper bound: 0.3565495
time: 4.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9364443, 0.9367080
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1576171, 1.1603527
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9588950, 0.9583623
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3787003, 1.3758082
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1516294, 1.1585712
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8957257, 0.8951089
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5471010, 1.5543704
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1145105, 1.1188006
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0065136, 0.9988933
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9964447, 0.9965570

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.44 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3570352, upper bound: 0.3606476
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3614344, upper bound: 0.3562489
time: 4.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9367080, 0.9364443
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1603532, 1.1576166
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9583623, 0.9588950
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3758087, 1.3787003
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1585712, 1.1516294
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8951087, 0.8957255
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5543709, 1.5471010
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1188006, 1.1145105
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9988933, 1.0065136
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9965572, 0.9964449

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.55 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3562479, upper bound: 0.3614345
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3606464, upper bound: 0.3570362
time: 4.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9367085, 0.9364445
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1603551, 1.1576195
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9583671, 0.9588902
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3758097, 1.3787012
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1585727, 1.1516309
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8951125, 0.8957291
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5543747, 1.5471058
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1188016, 1.1145120
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9988956, 1.0065160
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9965515, 0.9964504

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3565484, upper bound: 0.3614363
time: 5.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3609470, upper bound: 0.3570370
time: 4.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9378767, 0.9352756
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1596856, 1.1582842
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9591749, 0.9580824
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3785019, 1.3760066
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1538863, 1.1563144
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8952680, 0.8955667
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5474205, 1.5540504
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1137414, 1.1195693
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0063076, 0.9990995
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9957609, 0.9972408

Time for backsubstitution: 23.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3570443, upper bound: 0.3606385
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3614435, upper bound: 0.3562398
time: 4.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9378772, 0.9352760
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1596885, 1.1582861
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9591796, 0.9580777
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3785028, 1.3760080
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1538877, 1.1563158
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8952718, 0.8955703
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5474253, 1.5540552
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1137428, 1.1195707
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0063100, 0.9991016
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9957552, 0.9972465

Time for backsubstitution: 22.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3573449, upper bound: 0.3606392
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3617441, upper bound: 0.3562405
time: 4.16 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3562395, upper bound: 0.3617452
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3606381, upper bound: 0.3573459
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3562387, upper bound: 0.3614429
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3606373, upper bound: 0.3570436
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3570359, upper bound: 0.3609468
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3614352, upper bound: 0.3565495
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3570352, upper bound: 0.3606476
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3614344, upper bound: 0.3562489
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3562479, upper bound: 0.3614345
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3606464, upper bound: 0.3570362
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3565484, upper bound: 0.3614363
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3609470, upper bound: 0.3570370
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3570443, upper bound: 0.3606385
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3614435, upper bound: 0.3562398
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3573449, upper bound: 0.3606392
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.30
Output dim: 2, lower bound: -0.3617441, upper bound: 0.3562405

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9322004, 0.9306798
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1138763, 1.1030579
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9335039, 0.9393690
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3661718, 1.3193307
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1217360, 1.1399889
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8976738, 0.8904252
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5529423, 1.5485716
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1160603, 1.1121893
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9949465, 1.0037391
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9755573, 0.9747856

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3434327, upper bound: 0.3551417
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3499125, upper bound: 0.3500431
time: 5.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9306798, 0.9322007
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1030579, 1.1138763
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9393690, 0.9335036
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3193307, 1.3661718
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1399889, 1.1217356
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8904254, 0.8976738
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5485716, 1.5529423
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1121893, 1.1160603
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0037394, 0.9949467
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9747858, 0.9755573

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2857

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3500434, upper bound: 0.3499137
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3551405, upper bound: 0.3434338
time: 4.23 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.74
Output dim: 2, lower bound: -0.3434327, upper bound: 0.3551417
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.74
Output dim: 2, lower bound: -0.3499125, upper bound: 0.3500431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.74
Output dim: 2, lower bound: -0.3500434, upper bound: 0.3499137
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.74
Output dim: 2, lower bound: -0.3551405, upper bound: 0.3434338

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 58.02 + 517.36 = 575.38 seconds
