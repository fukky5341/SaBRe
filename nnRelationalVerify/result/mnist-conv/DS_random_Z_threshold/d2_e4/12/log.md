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
execution time: IAR + RelationalAnalysis = 22.69 + 35.70 = 58.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3728446, upper bound: 0.3728441

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720432, upper bound: 0.3720430
time: 7.78 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3720426
time: 4.82 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.61 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.61
Output dim: 2, lower bound: -0.3720432, upper bound: 0.3720430
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.61
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3720426

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9408398, 0.9420087
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1663098, 1.1656432
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9591858, 0.9599981
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3765931, 1.3792863
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1650543, 1.1603694
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8969035, 0.8970621
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5553265, 1.5483775
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1217823, 1.1167231
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9997163, 1.0071309
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9992242, 0.9984286

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3718561, upper bound: 0.3728101
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720091, upper bound: 0.3726557
time: 4.44 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9420090, 0.9408400
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1656432, 1.1663103
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9599984, 0.9591856
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3792863, 1.3765931
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1603694, 1.1650548
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8970623, 0.8969033
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5483770, 1.5553269
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1167231, 1.1217818
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -1.0071311, 0.9997165
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9984288, 0.9992244

Time for backsubstitution: 20.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 470

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717366
time: 4.32 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.51 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 2, lower bound: -0.3718561, upper bound: 0.3728101
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 2, lower bound: -0.3720091, upper bound: 0.3726557
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717366

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9408393, 0.9420075
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1663060, 1.1656370
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9591773, 0.9599943
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3765907, 1.3792853
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1650515, 1.1603651
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8968973, 0.8970594
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5553188, 1.5483651
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1217799, 1.1167221
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9997125, 1.0071292
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9992204, 0.9984186

Time for backsubstitution: 20.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 470

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3716998, upper bound: 0.3728083
time: 5.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717082, upper bound: 0.3724986
time: 5.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9408388, 0.9420080
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1663036, 1.1656394
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9591820, 0.9599898
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3765922, 1.3792844
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1650500, 1.1603665
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8969011, 0.8970559
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5553141, 1.5483694
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1217813, 1.1167207
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9997149, 1.0071268
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9992146, 0.9984243

Time for backsubstitution: 21.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 470

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3716991, upper bound: 0.3725093
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720088, upper bound: 0.3725008
time: 3.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3724998, upper bound: 0.3720099
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3724990, upper bound: 0.3717083
time: 4.99 seconds

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

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725081, upper bound: 0.3716989
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728087, upper bound: 0.3716996
time: 6.13 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.79 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.79
Output dim: 2, lower bound: -0.3716998, upper bound: 0.3728083
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.79
Output dim: 2, lower bound: -0.3717082, upper bound: 0.3724986
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.79
Output dim: 2, lower bound: -0.3716991, upper bound: 0.3725093
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.79
Output dim: 2, lower bound: -0.3720088, upper bound: 0.3725008
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.79
Output dim: 2, lower bound: -0.3724998, upper bound: 0.3720099
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.79
Output dim: 2, lower bound: -0.3724990, upper bound: 0.3717083
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.79
Output dim: 2, lower bound: -0.3725081, upper bound: 0.3716989
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.79
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

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1699

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3621013, upper bound: 0.3705075
time: 5.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3692497, upper bound: 0.3624679
time: 5.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 23.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 669

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3716118, upper bound: 0.3717423
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3709534, upper bound: 0.3724001
time: 5.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 22.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 2819

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3709705, upper bound: 0.3668571
time: 5.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3688097, upper bound: 0.3718383
time: 3.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 22.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 914

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3703649, upper bound: 0.3708522
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3703739, upper bound: 0.3708402
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 23.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 695

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 330

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3724297, upper bound: 0.3597551
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3602464, upper bound: 0.3719398
time: 3.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 22.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1704

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3615381, upper bound: 0.3696111
time: 6.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3704023, upper bound: 0.3607519
time: 7.78 seconds

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

Time for backsubstitution: 23.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 962

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3724546, upper bound: 0.3606917
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3614969, upper bound: 0.3716467
time: 4.18 seconds

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

Time for backsubstitution: 23.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 1699
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2857

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3718629, upper bound: 0.3703074
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3714165, upper bound: 0.3707553
time: 4.78 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3621013, upper bound: 0.3705075
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3692497, upper bound: 0.3624679
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3716118, upper bound: 0.3717423
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3709534, upper bound: 0.3724001
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3709705, upper bound: 0.3668571
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3688097, upper bound: 0.3718383
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3703649, upper bound: 0.3708522
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3703739, upper bound: 0.3708402
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3724297, upper bound: 0.3597551
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3602464, upper bound: 0.3719398
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3615381, upper bound: 0.3696111
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3704023, upper bound: 0.3607519
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3724546, upper bound: 0.3606917
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3614969, upper bound: 0.3716467
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3718629, upper bound: 0.3703074
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.99
Output dim: 2, lower bound: -0.3714165, upper bound: 0.3707553

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9327533, 0.9337511
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1519151, 1.1514206
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9493687, 0.9544966
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3735309, 1.3747902
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1531539, 1.1499524
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8936744, 0.8921475
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5543900, 1.5444789
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1192546, 1.1133575
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9982724, 1.0057864
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9963827, 0.9947274

Time for backsubstitution: 23.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 2319

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 599

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3618227, upper bound: 0.3703490
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3619416, upper bound: 0.3702287
time: 4.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9311502, 0.9352219
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1500182, 1.1531029
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9530694, 0.9504704
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3722954, 1.3758788
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1523170, 1.1507254
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8923078, 0.8933759
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5511084, 1.5475187
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1191759, 1.1134262
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9985623, 1.0054803
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9962015, 0.9948921

Time for backsubstitution: 22.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 912
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 2319
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 2857
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 1940
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 898
type: DSZ, layer: 3, pos: 569
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 2069
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 2318
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2220
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 2921
type: DSZ, layer: 3, pos: 695
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 969

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 912

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3154397, upper bound: 0.3086603
time: 5.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3154397, upper bound: 0.3086603
time: 5.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.5376854, -6.5594296, -8.5376854, -6.5594296, -0.9367242, 0.9364614
1: -13.8911514, -11.7880459, -13.8911514, -11.7880459, -1.1602707, 1.1575341
2: 7.0836673, 8.8593063, 7.0836673, 8.8593063, -0.9583151, 0.9588456
3: -4.7166414, -3.1604991, -4.7166414, -3.1604991, -1.3757458, 1.3786411
4: -10.4623423, -8.5371161, -10.4623423, -8.5371161, -1.1585851, 1.1516409
5: -10.2437754, -8.5587864, -10.2437754, -8.5587864, -0.8951232, 0.8957400
6: -12.8471022, -10.2882442, -12.8471022, -10.2882442, -1.5544343, 1.5471463
7: -3.9641032, -2.4725952, -3.9641032, -2.4725952, -1.1187787, 1.1144848
8: -1.6362410, -0.2654872, -1.6362410, -0.2654872, -0.9988508, 1.0064759
9: -8.8381767, -6.9597278, -8.8381767, -6.9597278, -0.9965568, 0.9964447

Time for backsubstitution: 22.98 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.38 + 544.16 = 602.55 seconds
