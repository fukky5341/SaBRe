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
execution time: IAR + RelationalAnalysis = 21.37 + 32.52 = 53.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0667589, upper bound: 0.0667588

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 612
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 612

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0636663, upper bound: 0.0667575
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667576, upper bound: 0.0636661
time: 2.51 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.51 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.51
Output dim: 9, lower bound: -0.0636663, upper bound: 0.0667575
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.51
Output dim: 9, lower bound: -0.0667576, upper bound: 0.0636661

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2007201, 0.2016404
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1387684, 0.1390884
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1127617, 0.1126598
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1333054, 0.1311638
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1151326, 0.1151549
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1067718, 0.1067872
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1640067, 0.1642769
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1315455, 0.1316773
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1140929, 0.1133886
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0977594, 0.0977664

Time for backsubstitution: 7.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0632423, upper bound: 0.0661185
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0630267, upper bound: 0.0663327
time: 2.52 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2016401, 0.2007201
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1390884, 0.1387684
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1126597, 0.1127616
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1311637, 0.1333055
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1151550, 0.1151327
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1067872, 0.1067718
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1642770, 0.1640067
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1316773, 0.1315455
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1133885, 0.1140930
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0977664, 0.0977594

Time for backsubstitution: 6.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0663332, upper bound: 0.0630265
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0661184, upper bound: 0.0632415
time: 2.52 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 12.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.12
Output dim: 9, lower bound: -0.0632423, upper bound: 0.0661185
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.12
Output dim: 9, lower bound: -0.0630267, upper bound: 0.0663327
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.12
Output dim: 9, lower bound: -0.0663332, upper bound: 0.0630265
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.12
Output dim: 9, lower bound: -0.0661184, upper bound: 0.0632415

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1683008, 0.1586800
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1263266, 0.1232555
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1007519, 0.0967694
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0954055, 0.0964830
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0913938, 0.0860334
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1034532, 0.1027757
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1490768, 0.1447268
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1200602, 0.1207224
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1014614, 0.1013834
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935122, 0.0935621

Time for backsubstitution: 7.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0613349, upper bound: 0.0643767
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0615003, upper bound: 0.0642107
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1577597, 0.1692210
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1229355, 0.1266465
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0968714, 0.1006500
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0986247, 0.0932638
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0860112, 0.0914161
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1027603, 0.1034686
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1444566, 0.1493470
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1205907, 0.1201919
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1020878, 0.1007570
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935551, 0.0935192

Time for backsubstitution: 7.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0611196, upper bound: 0.0645914
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0612852, upper bound: 0.0644254
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1692210, 0.1577597
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1266465, 0.1229355
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1006500, 0.0968714
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0932638, 0.0986247
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0914161, 0.0860112
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1034686, 0.1027603
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1493469, 0.1444566
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1201919, 0.1205907
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1007570, 0.1020878
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935192, 0.0935551

Time for backsubstitution: 7.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0644261, upper bound: 0.0612850
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645916, upper bound: 0.0611193
time: 2.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1586800, 0.1683008
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1232555, 0.1263266
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0967694, 0.1007519
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0964830, 0.0954055
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0860334, 0.0913938
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1027757, 0.1034531
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1447268, 0.1490768
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1207224, 0.1200602
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1013834, 0.1014614
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935620, 0.0935122

Time for backsubstitution: 7.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0642112, upper bound: 0.0615004
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0643766, upper bound: 0.0613341
time: 2.49 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 12.40 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0613349, upper bound: 0.0643767
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0615003, upper bound: 0.0642107
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0611196, upper bound: 0.0645914
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0612852, upper bound: 0.0644254
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0644261, upper bound: 0.0612850
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0645916, upper bound: 0.0611193
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0642112, upper bound: 0.0615004
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.40
Output dim: 9, lower bound: -0.0643766, upper bound: 0.0613341

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1676800, 0.1584611
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1232647, 0.1205050
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0982347, 0.0932621
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0945224, 0.0968449
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0906433, 0.0846615
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1017742, 0.1014917
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1448783, 0.1392325
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1161817, 0.1168592
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1014031, 0.1013543
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0913247, 0.0916715

Time for backsubstitution: 7.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0613285, upper bound: 0.0638387
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0607941, upper bound: 0.0643701
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1683008, 0.1580592
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1263266, 0.1201936
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1007519, 0.0942522
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0954055, 0.0955999
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0913938, 0.0852828
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1021691, 0.1027757
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1490768, 0.1405283
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1161971, 0.1207224
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1014324, 0.1013834
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935122, 0.0913746

Time for backsubstitution: 7.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0614947, upper bound: 0.0636738
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0609601, upper bound: 0.0642046
time: 2.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1571389, 0.1690022
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1198736, 0.1238962
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0943542, 0.0971426
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0977415, 0.0936257
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0852606, 0.0900442
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1010814, 0.1021845
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1402581, 0.1438527
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1167121, 0.1163287
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1020295, 0.1007280
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0913676, 0.0916286

Time for backsubstitution: 7.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0611139, upper bound: 0.0640512
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0605825, upper bound: 0.0645855
time: 2.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1577597, 0.1686003
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1229355, 0.1235847
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0968714, 0.0981327
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0986247, 0.0923807
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0860112, 0.0906655
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1014762, 0.1034686
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1444566, 0.1451485
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1167275, 0.1201919
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1020587, 0.1007570
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935551, 0.0913317

Time for backsubstitution: 8.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0612798, upper bound: 0.0638859
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0607479, upper bound: 0.0644196
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1686003, 0.1575409
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1235846, 0.1201850
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0981327, 0.0933639
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0923807, 0.0989866
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0906655, 0.0846379
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1017897, 0.1014762
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1451485, 0.1389616
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1163133, 0.1167275
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1006987, 0.1020587
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0913317, 0.0916644

Time for backsubstitution: 8.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0644200, upper bound: 0.0607476
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638861, upper bound: 0.0612795
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1692210, 0.1571389
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1266465, 0.1198736
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1006500, 0.0943542
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0932638, 0.0977415
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0914161, 0.0852606
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1021845, 0.1027603
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1493469, 0.1402581
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1163287, 0.1205907
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1007280, 0.1020878
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935192, 0.0913676

Time for backsubstitution: 7.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645855, upper bound: 0.0605819
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640512, upper bound: 0.0611133
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1580592, 0.1680819
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1201936, 0.1235762
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0942522, 0.0972445
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0955999, 0.0957674
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0852828, 0.0900206
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1010969, 0.1021691
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1405283, 0.1435817
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1168438, 0.1161971
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1013251, 0.1014324
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0913745, 0.0916216

Time for backsubstitution: 7.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0642052, upper bound: 0.0609596
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0636740, upper bound: 0.0614945
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1586800, 0.1676800
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1232555, 0.1232647
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0967694, 0.0982347
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0964830, 0.0945224
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0860334, 0.0906433
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1014917, 0.1034531
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1447268, 0.1448783
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1168592, 0.1200602
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1013543, 0.1014614
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0935620, 0.0913247

Time for backsubstitution: 8.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2251
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0643706, upper bound: 0.0607939
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638395, upper bound: 0.0613285
time: 2.88 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0613285, upper bound: 0.0638387
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0607941, upper bound: 0.0643701
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0614947, upper bound: 0.0636738
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0609601, upper bound: 0.0642046
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0611139, upper bound: 0.0640512
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0605825, upper bound: 0.0645855
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0612798, upper bound: 0.0638859
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0607479, upper bound: 0.0644196
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0644200, upper bound: 0.0607476
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0638861, upper bound: 0.0612795
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0645855, upper bound: 0.0605819
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0640512, upper bound: 0.0611133
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0642052, upper bound: 0.0609596
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0636740, upper bound: 0.0614945
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0643706, upper bound: 0.0607939
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.09
Output dim: 9, lower bound: -0.0638395, upper bound: 0.0613285

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1671795, 0.1580877
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1226965, 0.1199804
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0981667, 0.0932059
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0940969, 0.0965053
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0903513, 0.0843829
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1015307, 0.1013099
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1445645, 0.1389866
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1158699, 0.1164684
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1010281, 0.1009657
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0912082, 0.0915235

Time for backsubstitution: 8.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0584859, upper bound: 0.0638024
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0612919, upper bound: 0.0603263
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1673066, 0.1579606
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1227404, 0.1199368
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0981785, 0.0931942
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0941838, 0.0964195
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0903647, 0.0843798
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1015925, 0.1012481
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1446441, 0.1389186
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1157917, 0.1165474
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1010145, 0.1009939
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0911767, 0.0915554

Time for backsubstitution: 8.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0579510, upper bound: 0.0643332
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0607574, upper bound: 0.0608576
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1678003, 0.1576857
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1257584, 0.1196689
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1006841, 0.0941960
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0949801, 0.0952602
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0911019, 0.0850042
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1019255, 0.1025940
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1487630, 0.1402824
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1158852, 0.1203316
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1010574, 0.1009947
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0933959, 0.0912266

Time for backsubstitution: 7.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0579819, upper bound: 0.0636366
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0614572, upper bound: 0.0608299
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1679274, 0.1575587
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1258023, 0.1196253
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1006958, 0.0941842
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0950670, 0.0951744
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0911152, 0.0850011
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1019873, 0.1025322
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1488426, 0.1402145
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1158070, 0.1204106
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1010437, 0.1010229
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0933645, 0.0912584

Time for backsubstitution: 8.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0574474, upper bound: 0.0641679
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0609229, upper bound: 0.0613614
time: 2.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1566384, 0.1686287
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1193054, 0.1233718
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0942862, 0.0970864
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0973161, 0.0932871
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0849788, 0.0897656
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1008379, 0.1020027
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1399443, 0.1436185
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1164002, 0.1159387
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1016691, 0.1003393
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0912515, 0.0914806

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0582706, upper bound: 0.0640141
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0610768, upper bound: 0.0605380
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1567655, 0.1685017
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1193489, 0.1233279
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0942980, 0.0970747
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0974019, 0.0932003
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0849820, 0.0897523
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1008997, 0.1019409
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1400122, 0.1435389
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1163213, 0.1160169
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1016408, 0.1003530
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0912197, 0.0915121

Time for backsubstitution: 8.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577393, upper bound: 0.0645487
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0605455, upper bound: 0.0610721
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1572592, 0.1682268
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1223673, 0.1230603
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0968035, 0.0980765
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0981993, 0.0920421
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0857294, 0.0903869
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1012327, 0.1032868
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1441428, 0.1449144
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1164157, 0.1198019
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1016983, 0.1003683
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0934392, 0.0911838

Time for backsubstitution: 8.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577670, upper bound: 0.0638486
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0612424, upper bound: 0.0610418
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1573863, 0.1680997
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1224109, 0.1230164
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0968153, 0.0980648
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0982851, 0.0919552
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0857325, 0.0903736
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1012945, 0.1032250
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1442107, 0.1448347
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1163368, 0.1198801
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1016701, 0.1003820
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0934074, 0.0912153

Time for backsubstitution: 8.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0572355, upper bound: 0.0643828
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0607109, upper bound: 0.0615760
time: 2.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1680997, 0.1571674
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1230165, 0.1196604
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0980648, 0.0933077
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0919552, 0.0986469
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0903736, 0.0843593
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1015461, 0.1012945
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1448347, 0.1387157
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1160015, 0.1163368
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1003238, 0.1016701
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0912153, 0.0915165

Time for backsubstitution: 9.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0615764, upper bound: 0.0607107
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0643826, upper bound: 0.0572348
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1682268, 0.1570404
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1230603, 0.1196168
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0980766, 0.0932960
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0920420, 0.0985611
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0903869, 0.0843562
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1016079, 0.1012327
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1449144, 0.1386478
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1159233, 0.1164157
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1003101, 0.1016983
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0911838, 0.0915483

Time for backsubstitution: 8.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0610421, upper bound: 0.0612420
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638488, upper bound: 0.0577665
time: 2.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1687205, 0.1567655
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1260784, 0.1193489
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1005821, 0.0942980
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0928384, 0.0974019
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0911241, 0.0849820
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1019410, 0.1025786
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1490332, 0.1400122
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1160168, 0.1201999
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1003530, 0.1016991
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0934029, 0.0912197

Time for backsubstitution: 8.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0610725, upper bound: 0.0605453
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645482, upper bound: 0.0577385
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1688476, 0.1566384
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1261222, 0.1193054
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1005938, 0.0942862
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0929253, 0.0973161
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0911375, 0.0849788
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1020027, 0.1025168
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1491128, 0.1399443
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1159386, 0.1202788
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1003393, 0.1017273
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0933714, 0.0912515

Time for backsubstitution: 9.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0605385, upper bound: 0.0610766
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640142, upper bound: 0.0582703
time: 2.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1575586, 0.1677085
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1196253, 0.1230519
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0941842, 0.0971883
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0951744, 0.0954288
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0850011, 0.0897420
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1008533, 0.1019873
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1402145, 0.1433476
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1165320, 0.1158071
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1009647, 0.1010437
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0912584, 0.0914736

Time for backsubstitution: 8.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0613616, upper bound: 0.0609228
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0641682, upper bound: 0.0574471
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1576857, 0.1675814
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1196690, 0.1230080
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0941960, 0.0971765
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0952602, 0.0953420
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0850042, 0.0897287
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1009151, 0.1019255
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1402824, 0.1432679
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1164531, 0.1158853
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1009364, 0.1010574
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0912266, 0.0915052

Time for backsubstitution: 8.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0608303, upper bound: 0.0614567
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0636365, upper bound: 0.0579815
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1581795, 0.1673066
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1226872, 0.1227404
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0967016, 0.0981785
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0960576, 0.0941838
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0857516, 0.0903647
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1012481, 0.1032714
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1444130, 0.1446441
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1165473, 0.1196703
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1009939, 0.1010727
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0934461, 0.0911767

Time for backsubstitution: 9.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0608577, upper bound: 0.0607574
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0643337, upper bound: 0.0579510
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1583065, 0.1671795
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1227309, 0.1226965
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0967133, 0.0981668
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0961434, 0.0940969
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0857548, 0.0903513
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1013100, 0.1032096
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1444809, 0.1445645
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1164684, 0.1197485
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1009657, 0.1010864
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0934143, 0.0912082

Time for backsubstitution: 8.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0603263, upper bound: 0.0612915
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0638023, upper bound: 0.0584856
time: 2.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0584859, upper bound: 0.0638024
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0612919, upper bound: 0.0603263
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0579510, upper bound: 0.0643332
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0607574, upper bound: 0.0608576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0579819, upper bound: 0.0636366
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0614572, upper bound: 0.0608299
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0574474, upper bound: 0.0641679
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0609229, upper bound: 0.0613614
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0582706, upper bound: 0.0640141
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0610768, upper bound: 0.0605380
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0577393, upper bound: 0.0645487
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0605455, upper bound: 0.0610721
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0577670, upper bound: 0.0638486
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0612424, upper bound: 0.0610418
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0572355, upper bound: 0.0643828
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0607109, upper bound: 0.0615760
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0615764, upper bound: 0.0607107
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0643826, upper bound: 0.0572348
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0610421, upper bound: 0.0612420
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0638488, upper bound: 0.0577665
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0610725, upper bound: 0.0605453
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0645482, upper bound: 0.0577385
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0605385, upper bound: 0.0610766
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0640142, upper bound: 0.0582703
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0613616, upper bound: 0.0609228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0641682, upper bound: 0.0574471
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0608303, upper bound: 0.0614567
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0636365, upper bound: 0.0579815
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0608577, upper bound: 0.0607574
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0643337, upper bound: 0.0579510
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0603263, upper bound: 0.0612915
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 9, lower bound: -0.0638023, upper bound: 0.0584856

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1664215, 0.1576827
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1223578, 0.1195659
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0979428, 0.0931190
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0939444, 0.0964065
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0893062, 0.0831024
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1015260, 0.1013058
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1439130, 0.1381893
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1157647, 0.1161994
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1007299, 0.1008290
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0901654, 0.0910456

Time for backsubstitution: 8.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0482811, upper bound: 0.0633439
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0580218, upper bound: 0.0538467
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1665486, 0.1575556
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1224017, 0.1195223
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0979545, 0.0931073
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0940313, 0.0963207
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0893195, 0.0830992
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1015878, 0.1012440
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1439926, 0.1381213
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1156865, 0.1162784
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1007162, 0.1008572
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0901339, 0.0910776

Time for backsubstitution: 7.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0477469, upper bound: 0.0638754
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0574874, upper bound: 0.0543778
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1670857, 0.1572807
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1254198, 0.1192544
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1004730, 0.0939310
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0948277, 0.0951615
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0900567, 0.0837237
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1019209, 0.1025899
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1481115, 0.1394851
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1157799, 0.1200781
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1007591, 0.1008580
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0924128, 0.0907502

Time for backsubstitution: 7.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0477767, upper bound: 0.0633098
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0575177, upper bound: 0.0535012
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1672128, 0.1571537
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1254637, 0.1192109
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1004847, 0.0939193
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0949145, 0.0950757
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0900701, 0.0837205
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1019827, 0.1025281
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1481911, 0.1394172
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1157017, 0.1201570
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1007454, 0.1008862
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0923814, 0.0907820

Time for backsubstitution: 7.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0472431, upper bound: 0.0638420
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0569839, upper bound: 0.0540326
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1558805, 0.1682237
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1189667, 0.1229573
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0940623, 0.0969995
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0971636, 0.0931884
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0839336, 0.0884851
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1008332, 0.1019986
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1392928, 0.1428212
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1162952, 0.1156697
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1013708, 0.1002026
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0902085, 0.0910028

Time for backsubstitution: 7.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0480663, upper bound: 0.0635558
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0578071, upper bound: 0.0540587
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1560075, 0.1680967
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1190103, 0.1229134
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0940740, 0.0969878
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0972494, 0.0931015
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0839368, 0.0884717
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1008950, 0.1019368
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1393607, 0.1427415
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1162162, 0.1157479
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1013426, 0.1002163
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0901767, 0.0910343

Time for backsubstitution: 8.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0475342, upper bound: 0.0640897
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0572752, upper bound: 0.0545927
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1565447, 0.1678218
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1220287, 0.1226459
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0965924, 0.0978116
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0980468, 0.0919433
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0846842, 0.0891064
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1012280, 0.1032827
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1434913, 0.1441170
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1163104, 0.1195483
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1014000, 0.1002316
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0924560, 0.0907073

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0475623, upper bound: 0.0635225
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0573030, upper bound: 0.0537135
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1566717, 0.1676948
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1220722, 0.1226020
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0966041, 0.0977998
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0981326, 0.0918565
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0846874, 0.0890930
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1012899, 0.1032209
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1435592, 0.1440374
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1162315, 0.1196265
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1013718, 0.1002453
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0924242, 0.0907388

Time for backsubstitution: 7.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0470307, upper bound: 0.0640564
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0567719, upper bound: 0.0542480
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1676947, 0.1564095
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1226020, 0.1193217
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0977998, 0.0930838
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0918565, 0.0984944
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0890930, 0.0833142
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1015420, 0.1012898
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1440374, 0.1380641
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1157325, 0.1162315
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1001870, 0.1013718
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0907388, 0.0904737

Time for backsubstitution: 7.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0542477, upper bound: 0.0567714
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640567, upper bound: 0.0470307
time: 2.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.1678218, 0.1562824
1: -12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1226459, 0.1192781
2: -8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.0978115, 0.0930721
3: -8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.0919433, 0.0984086
4: -3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.0891064, 0.0833111
5: -5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1016038, 0.1012280
6: -13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1441170, 0.1379961
7: -3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1156543, 0.1163105
8: -1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1001734, 0.1014000
9: 2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0907073, 0.0905055

Time for backsubstitution: 7.76 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 53.89 + 546.15 = 600.04 seconds
