## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06342095499999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2105641, 0.2105641)
1: (-12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1409148, 0.1409148)
2: (-8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1139829, 0.1139829)
3: (-8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1432159, 0.1432159)
4: (-3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1160847, 0.1160847)
5: (-5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1068949, 0.1068948)
6: (-13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1659200, 0.1659200)
7: (-3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1338000, 0.1338000)
8: (-1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1171519, 0.1171519)
9: (2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0977726, 0.0977726)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.39 + 32.75 = 55.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0667589, upper bound: 0.0667588

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667529, upper bound: 0.0662242
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0662244, upper bound: 0.0667523
time: 2.51 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.99 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.99
Output dim: 9, lower bound: -0.0667529, upper bound: 0.0662242
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.99
Output dim: 9, lower bound: -0.0662244, upper bound: 0.0667523

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2103643, 0.2104917
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1406704, 0.1407139
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1139562, 0.1139680
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1429893, 0.1430761
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1159333, 0.1159364
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1067594, 0.1068213
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1657946, 0.1658626
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1336747, 0.1335958
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1170095, 0.1169958
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0977200, 0.0976882

Time for backsubstitution: 8.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0566932, upper bound: 0.0658977
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664265, upper bound: 0.0561644
time: 2.53 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2104917, 0.2103643
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1407139, 0.1406704
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1139680, 0.1139562
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1430761, 0.1429893
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1159364, 0.1159333
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1068212, 0.1067594
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1658626, 0.1657946
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1335958, 0.1336747
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1169958, 0.1170095
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0976882, 0.0977200

Time for backsubstitution: 7.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638221, upper bound: 0.0667158
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0661879, upper bound: 0.0643503
time: 2.40 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 12.62 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.62
Output dim: 9, lower bound: -0.0566932, upper bound: 0.0658977
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.62
Output dim: 9, lower bound: -0.0664265, upper bound: 0.0561644
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.62
Output dim: 9, lower bound: -0.0638221, upper bound: 0.0667158
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.62
Output dim: 9, lower bound: -0.0661879, upper bound: 0.0643503

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2038401, 0.2039323
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1299095, 0.1299005
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1000217, 0.1039388
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1377823, 0.1379491
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1158862, 0.1158830
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1057323, 0.1054262
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1650720, 0.1650007
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1102859, 0.1036240
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1093978, 0.1102423
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0618396, 0.0710875

Time for backsubstitution: 7.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0543907, upper bound: 0.0658613
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0566569, upper bound: 0.0634958
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2038053, 0.2039671
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1298569, 0.1299530
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1039271, 0.1000334
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1378622, 0.1378692
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1158798, 0.1158892
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1053644, 0.1057941
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1649327, 0.1651398
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1037030, 0.1102070
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1102559, 0.1093841
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0711193, 0.0618078

Time for backsubstitution: 8.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0633283, upper bound: 0.0561631
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664250, upper bound: 0.0528228
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2097771, 0.2100031
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1403757, 0.1402563
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1137569, 0.1137041
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1429241, 0.1428909
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1148918, 0.1146533
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1068166, 0.1067553
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1652113, 0.1649975
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1335059, 0.1334211
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1166979, 0.1168731
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0967048, 0.0973032

Time for backsubstitution: 7.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0633925, upper bound: 0.0660773
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0631801, upper bound: 0.0662921
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2101302, 0.2096500
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1402998, 0.1403321
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1137159, 0.1137451
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1429778, 0.1428372
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1146564, 0.1148887
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1068171, 0.1067548
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1650655, 0.1651434
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1333421, 0.1335849
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1168595, 0.1167116
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0972713, 0.0967367

Time for backsubstitution: 8.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0561282, upper bound: 0.0640241
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0658618, upper bound: 0.0543909
time: 2.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.41 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0543907, upper bound: 0.0658613
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0566569, upper bound: 0.0634958
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0633283, upper bound: 0.0561631
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0664250, upper bound: 0.0528228
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0633925, upper bound: 0.0660773
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0631801, upper bound: 0.0662921
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0561282, upper bound: 0.0640241
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.41
Output dim: 9, lower bound: -0.0658618, upper bound: 0.0543909

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2031258, 0.2035708
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1295713, 0.1294864
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0998105, 0.1036867
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1376305, 0.1378510
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1148415, 0.1146032
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1057276, 0.1054220
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1644206, 0.1642035
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1101961, 0.1033705
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1090999, 0.1101060
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0608564, 0.0706707

Time for backsubstitution: 7.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0520464, upper bound: 0.0639872
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0515428, upper bound: 0.0639540
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2034787, 0.2032180
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1294954, 0.1295623
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0997695, 0.1037277
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1376842, 0.1377972
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1146061, 0.1148385
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1057281, 0.1054215
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1642748, 0.1643493
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1100323, 0.1035343
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1092615, 0.1099444
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0614229, 0.0701043

Time for backsubstitution: 7.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0546732, upper bound: 0.0605104
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0550186, upper bound: 0.0610148
time: 2.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1923157, 0.1915574
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1257356, 0.1255118
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1011306, 0.0973389
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1251975, 0.1273463
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1149291, 0.1149163
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1052483, 0.1056626
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1629514, 0.1628883
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0990593, 0.1054316
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1062518, 0.1060844
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0710926, 0.0617741

Time for backsubstitution: 8.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0660012, upper bound: 0.0521809
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0657864, upper bound: 0.0523932
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1770715, 0.1667563
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1276157, 0.1241049
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1017057, 0.0977724
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1048265, 0.1080126
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0910232, 0.0854020
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1033898, 0.1026357
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1501077, 0.1452620
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1218407, 0.1222856
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1038338, 0.1046500
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0923946, 0.0930357

Time for backsubstitution: 8.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0603004, upper bound: 0.0660761
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0633907, upper bound: 0.0629842
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1665304, 0.1772974
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1242243, 0.1274961
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0978251, 0.1016529
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1080446, 0.1047934
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0856405, 0.0907745
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1026970, 0.1033285
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1454758, 0.1498822
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1223705, 0.1217551
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1044602, 0.1040091
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0924374, 0.0929924

Time for backsubstitution: 8.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0608318, upper bound: 0.0645502
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0603272, upper bound: 0.0643840
time: 2.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2036058, 0.2030909
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1295390, 0.1295187
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0997813, 0.1037160
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1377711, 0.1377105
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1146095, 0.1148354
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1057899, 0.1053597
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1643428, 0.1642814
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1099534, 0.1036132
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1092478, 0.1099580
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0613910, 0.0701361

Time for backsubstitution: 8.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0556982, upper bound: 0.0633851
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0554862, upper bound: 0.0636001
time: 2.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2035710, 0.2031257
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1294864, 0.1295713
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1036867, 0.0998105
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1378510, 0.1376305
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1146030, 0.1148416
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1054220, 0.1057276
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1642035, 0.1644206
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1033704, 0.1101961
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1101060, 0.1090999
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0706707, 0.0608564

Time for backsubstitution: 8.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0627635, upper bound: 0.0543894
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0658604, upper bound: 0.0510494
time: 2.68 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0520464, upper bound: 0.0639872
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0515428, upper bound: 0.0639540
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0546732, upper bound: 0.0605104
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0550186, upper bound: 0.0610148
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0660012, upper bound: 0.0521809
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0657864, upper bound: 0.0523932
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0603004, upper bound: 0.0660761
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0633907, upper bound: 0.0629842
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0608318, upper bound: 0.0645502
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0603272, upper bound: 0.0643840
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0556982, upper bound: 0.0633851
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0554862, upper bound: 0.0636001
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0627635, upper bound: 0.0543894
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.67
Output dim: 9, lower bound: -0.0658604, upper bound: 0.0510494

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2024688, 0.2033157
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1265138, 0.1267405
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0972303, 0.1002943
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1367342, 0.1381997
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1140938, 0.1132340
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1040388, 0.1041281
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1602246, 0.1587109
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1060295, 0.0994169
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1090456, 0.1100810
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0585556, 0.0684015

Time for backsubstitution: 8.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0516226, upper bound: 0.0633455
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0514075, upper bound: 0.0635574
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2031258, 0.2029138
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1295713, 0.1264290
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0998105, 0.1011065
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1376305, 0.1369547
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1148415, 0.1138554
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1044336, 0.1054220
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1644206, 0.1600074
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1062425, 0.1033705
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1090749, 0.1101060
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0608564, 0.0683698

Time for backsubstitution: 7.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0511184, upper bound: 0.0633116
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0509038, upper bound: 0.0635240
time: 2.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1595952, 0.1482959
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1129696, 0.1093547
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0890788, 0.0814067
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0870985, 0.0924654
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0910499, 0.0856645
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1018214, 0.1015429
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1478330, 0.1431497
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0873870, 0.0942897
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0933873, 0.0938463
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0667815, 0.0575063

Time for backsubstitution: 8.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0635986, upper bound: 0.0521443
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0659648, upper bound: 0.0498793
time: 2.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1490541, 0.1588369
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1095785, 0.1127462
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0851983, 0.0852872
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0903177, 0.0892473
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0856774, 0.0910472
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1011286, 0.1022357
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1432128, 0.1477817
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0879174, 0.0937600
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0940282, 0.0932199
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0668248, 0.0574634

Time for backsubstitution: 8.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638789, upper bound: 0.0507553
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0639125, upper bound: 0.0504102
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1672128, 0.1578179
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1254637, 0.1222728
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1004847, 0.0964494
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0949145, 0.0959589
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0900701, 0.0844711
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1032668, 0.1025281
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1481911, 0.1436156
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1195805, 0.1201570
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1007744, 0.1008862
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0923814, 0.0930295

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0500914, upper bound: 0.0657497
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0599685, upper bound: 0.0560165
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1658735, 0.1770424
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1211663, 0.1247495
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0953050, 0.0983207
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1071692, 0.1051630
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0848933, 0.0894061
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1010182, 0.1020445
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1412794, 0.1443893
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1184934, 0.1178935
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1044061, 0.1039842
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0901901, 0.0910406

Time for backsubstitution: 7.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0508762, upper bound: 0.0640914
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0603730, upper bound: 0.0545944
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1665304, 0.1766405
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1242243, 0.1244381
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0978251, 0.0991327
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1080446, 0.1039179
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0856405, 0.0900274
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1014130, 0.1033285
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1454758, 0.1456858
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1185088, 0.1217551
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1044353, 0.1040091
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0924374, 0.0907451

Time for backsubstitution: 8.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0503723, upper bound: 0.0640579
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0598691, upper bound: 0.0542492
time: 2.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1603592, 0.1703854
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1133874, 0.1167582
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0838496, 0.0916647
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1028914, 0.0996126
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0853579, 0.0909564
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1016703, 0.1019329
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1446073, 0.1491662
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0988171, 0.0919465
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0970102, 0.0970940
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0571232, 0.0658250

Time for backsubstitution: 8.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0535029, upper bound: 0.0606152
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0538482, upper bound: 0.0611193
time: 2.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1920812, 0.1907159
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1253648, 0.1251296
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1008902, 0.0971160
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1251856, 0.1271070
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1136518, 0.1138679
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1053059, 0.1055961
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1622220, 0.1621689
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0987269, 0.1054208
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1061015, 0.1057998
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0706440, 0.0608227

Time for backsubstitution: 8.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0654304, upper bound: 0.0504105
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0652184, upper bound: 0.0506252
time: 2.60 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 13.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0516226, upper bound: 0.0633455
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0514075, upper bound: 0.0635574
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0511184, upper bound: 0.0633116
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0509038, upper bound: 0.0635240
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0635986, upper bound: 0.0521443
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0659648, upper bound: 0.0498793
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0638789, upper bound: 0.0507553
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0639125, upper bound: 0.0504102
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0500914, upper bound: 0.0657497
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0599685, upper bound: 0.0560165
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0508762, upper bound: 0.0640914
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0603730, upper bound: 0.0545944
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0503723, upper bound: 0.0640579
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0598691, upper bound: 0.0542492
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0535029, upper bound: 0.0606152
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0538482, upper bound: 0.0611193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0654304, upper bound: 0.0504105
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 9, lower bound: -0.0652184, upper bound: 0.0506252

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1592225, 0.1706106
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1103622, 0.1139802
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0812987, 0.0882432
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1018557, 0.1001021
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0848424, 0.0893653
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.0999192, 0.1007012
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1404892, 0.1436075
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0948932, 0.0877509
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0968226, 0.0972170
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0542878, 0.0640908

Time for backsubstitution: 7.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0480663, upper bound: 0.0635558
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0514057, upper bound: 0.0604588
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1598792, 0.1702087
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1134197, 0.1136687
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0838788, 0.0890553
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1027519, 0.0988570
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0855902, 0.0899866
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1003140, 0.1019952
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1446852, 0.1449040
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0951062, 0.0917045
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0968518, 0.0972419
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0565886, 0.0640591

Time for backsubstitution: 8.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0475623, upper bound: 0.0635225
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0509019, upper bound: 0.0604254
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1588807, 0.1479343
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1126310, 0.1089403
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0888677, 0.0811545
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0869461, 0.0923666
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0900047, 0.0843840
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1018168, 0.1015387
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1471815, 0.1423524
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0872972, 0.0940361
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0930890, 0.0937096
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0657983, 0.0570895

Time for backsubstitution: 8.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0611177, upper bound: 0.0505060
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0606139, upper bound: 0.0501608
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1592337, 0.1475813
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1125552, 0.1090161
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0888267, 0.0811955
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0869998, 0.0923129
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0897694, 0.0846194
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1018173, 0.1015382
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1470357, 0.1424983
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0871334, 0.0941999
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0932506, 0.0935480
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0663647, 0.0565230

Time for backsubstitution: 7.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640567, upper bound: 0.0470307
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640900, upper bound: 0.0475343
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1484332, 0.1586180
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1065163, 0.1099954
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0826209, 0.0817196
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0894136, 0.0895882
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0849262, 0.0896733
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.0994397, 0.1009416
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1390142, 0.1422866
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0839465, 0.0898046
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0939699, 0.0931908
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0645836, 0.0555191

Time for backsubstitution: 7.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0609029, upper bound: 0.0507181
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638415, upper bound: 0.0472424
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1490541, 0.1582160
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1095785, 0.1096839
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0851983, 0.0827098
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0903177, 0.0883432
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0856774, 0.0902960
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.0998345, 0.1022357
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1432128, 0.1435831
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0839620, 0.0937600
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0939991, 0.0932199
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0668248, 0.0552222

Time for backsubstitution: 8.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0603995, upper bound: 0.0503735
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638754, upper bound: 0.0477467
time: 2.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1581224, 0.1486926
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1124075, 0.1091641
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0850760, 0.0849462
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0890948, 0.0902189
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0900021, 0.0843969
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1022310, 0.1011244
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1471302, 0.1424155
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0936702, 0.0876638
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0929216, 0.0938916
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0564802, 0.0664080

Time for backsubstitution: 8.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0477469, upper bound: 0.0638754
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0472431, upper bound: 0.0638420
time: 2.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1593496, 0.1704836
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1104057, 0.1139363
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0813104, 0.0882314
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1019415, 0.1000152
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0848455, 0.0893520
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.0999810, 0.1006394
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1405571, 0.1435278
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0948142, 0.0878290
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0967944, 0.0972307
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0542559, 0.0641223

Time for backsubstitution: 8.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0475342, upper bound: 0.0640897
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0508744, upper bound: 0.0609932
time: 2.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1600063, 0.1700816
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1134633, 0.1136248
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0838906, 0.0890436
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1028377, 0.0987702
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0855933, 0.0899733
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1003758, 0.1019334
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1447531, 0.1448243
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0950272, 0.0917827
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0968236, 0.0972556
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0565568, 0.0640906

Time for backsubstitution: 7.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0470307, upper bound: 0.0640564
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0503706, upper bound: 0.0609597
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1593607, 0.1474543
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1125991, 0.1089725
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0888385, 0.0811838
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0870866, 0.0922271
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0897827, 0.0846162
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1018791, 0.1014764
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1471153, 0.1424303
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0870552, 0.0942788
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0932370, 0.0935762
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0663332, 0.0565549

Time for backsubstitution: 8.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0635224, upper bound: 0.0475617
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0635558, upper bound: 0.0480659
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1488197, 0.1579953
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1092076, 0.1123636
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0849580, 0.0850643
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0903047, 0.0890080
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0844000, 0.0899887
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1011863, 0.1021692
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1424834, 0.1470505
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0875849, 0.0937484
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0938634, 0.0929353
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0663761, 0.0565116

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0633102, upper bound: 0.0477767
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0633437, upper bound: 0.0482806
time: 2.55 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 13.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0480663, upper bound: 0.0635558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0514057, upper bound: 0.0604588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0475623, upper bound: 0.0635225
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0509019, upper bound: 0.0604254
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0611177, upper bound: 0.0505060
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0606139, upper bound: 0.0501608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0640567, upper bound: 0.0470307
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0640900, upper bound: 0.0475343
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0609029, upper bound: 0.0507181
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0638415, upper bound: 0.0472424
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0603995, upper bound: 0.0503735
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0638754, upper bound: 0.0477467
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0477469, upper bound: 0.0638754
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0472431, upper bound: 0.0638420
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0475342, upper bound: 0.0640897
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0508744, upper bound: 0.0609932
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0470307, upper bound: 0.0640564
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0503706, upper bound: 0.0609597
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0635224, upper bound: 0.0475617
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0635558, upper bound: 0.0480659
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0633102, upper bound: 0.0477767
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.58
Output dim: 9, lower bound: -0.0633437, upper bound: 0.0482806

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1467900, 0.1590983
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1059103, 0.1098483
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0785935, 0.0854361
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0913230, 0.0874275
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0838650, 0.0884102
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.0997875, 0.1005850
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1382318, 0.1416209
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.0900950, 0.0830843
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.0935179, 0.0932079
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0542539, 0.0640640

Time for backsubstitution: 8.44 seconds

No DS candidates found

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.13 + 454.40 = 509.54 seconds
