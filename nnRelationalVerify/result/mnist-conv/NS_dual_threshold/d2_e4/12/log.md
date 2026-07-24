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
execution time: IAR + RelationalAnalysis = 23.30 + 36.13 = 59.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3728446, upper bound: 0.3728441

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3720450
time: 5.30 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3728440
time: 4.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.25 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 10.25
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3720450
NS_B2, status: Status.UNKNOWN, split count: 1, time: 10.25
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3728440

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -8.5335140, -6.5600677, -8.5305357, -6.5644798, -0.9343023, 0.9359827
1: -13.8883953, -11.7894220, -13.8855553, -11.7912292, -1.1580944, 1.1580243
2: 7.0862308, 8.8571110, 7.0890179, 8.8523531, -0.9529982, 0.9547744
3: -4.7136874, -3.1635637, -4.7118793, -3.1690207, -1.3580136, 1.3601408
4: -10.4591522, -8.5467110, -10.4450283, -8.5531473, -1.1567507, 1.1450353
5: -10.2389441, -8.5593319, -10.2346210, -8.5627766, -0.8888724, 0.8883064
6: -12.8445501, -10.3106327, -12.8182182, -10.3268747, -1.5613427, 1.5507331
7: -3.9625266, -2.4840207, -3.9488356, -2.4919736, -1.1141915, 1.1041222
8: -1.6138616, -0.2685580, -1.5983412, -0.2951255, -1.0020385, 1.0143075
9: -8.8352432, -6.9607792, -8.8314934, -6.9639268, -0.9865789, 0.9867561

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717384
time: 4.64 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3720459
time: 4.60 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -8.5376835, -6.5594292, -8.5376806, -6.5594273, -0.9430444, 0.9420047
1: -13.8911476, -11.7880459, -13.8911495, -11.7880440, -1.1663070, 1.1645718
2: 7.0836678, 8.8593054, 7.0836720, 8.8593054, -0.9626939, 0.9599943
3: -4.7166386, -3.1605024, -4.7166386, -3.1605043, -1.3635778, 1.3792753
4: -10.4623394, -8.5371256, -10.4623375, -8.5371332, -1.1650448, 1.1713595
5: -10.2437725, -8.5587864, -10.2437716, -8.5587883, -0.8981812, 0.8970571
6: -12.8471031, -10.2882595, -12.8470974, -10.2882748, -1.5553217, 1.5937366
7: -3.9641006, -2.4726043, -3.9640999, -2.4726095, -1.1217756, 1.1281819
8: -1.6362171, -0.2654886, -1.6361995, -0.2654905, -1.0445223, 1.0071268
9: -8.8381767, -6.9597254, -8.8381748, -6.9597273, -0.9992185, 0.9930844

Time for backsubstitution: 21.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3728423
time: 12.28 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3728424
time: 5.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 39.63 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 39.63
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717384
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 39.63
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3720459
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 39.63
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3728423
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 39.63
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3728424

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -8.5289497, -6.5601254, -8.5229111, -6.5645790, -0.9296675, 0.9282961
1: -13.8846188, -11.7895842, -13.8792458, -11.7914963, -1.1506667, 1.1482806
2: 7.0871096, 8.8565035, 7.0904903, 8.8513222, -0.9510584, 0.9526589
3: -4.7132564, -3.1639099, -4.7111564, -3.1695971, -1.3557520, 1.3577981
4: -10.4512653, -8.5467319, -10.4318123, -8.5531816, -1.1488824, 1.1319022
5: -10.2382622, -8.5611544, -10.2334919, -8.5658197, -0.8851085, 0.8853085
6: -12.8437700, -10.3109322, -12.8169384, -10.3273640, -1.5595951, 1.5484972
7: -3.9623780, -2.4855578, -3.9485984, -2.4945509, -1.1107616, 1.1017885
8: -1.6135020, -0.2699442, -1.5977373, -0.2974353, -0.9994154, 1.0124180
9: -8.8351345, -6.9620018, -8.8313179, -6.9659829, -0.9824591, 0.9835176

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3717384
time: 4.63 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3717381
time: 5.95 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -8.5335083, -6.5600672, -8.5311069, -6.5565615, -0.9422848, 0.9323146
1: -13.8883905, -11.7894220, -13.8858433, -11.7845058, -1.1627154, 1.1545734
2: 7.0862303, 8.8571110, 7.0886574, 8.8543310, -0.9549093, 0.9542503
3: -4.7136879, -3.1635633, -4.7136364, -3.1674900, -1.3597441, 1.3603191
4: -10.4591465, -8.5467129, -10.4459009, -8.5405216, -1.1677444, 1.1391125
5: -10.2389450, -8.5593348, -10.2379408, -8.5622101, -0.8879652, 0.8916526
6: -12.8445501, -10.3106337, -12.8194294, -10.3248777, -1.5629797, 1.5510535
7: -3.9625273, -2.4840236, -3.9522784, -2.4910858, -1.1133475, 1.1071134
8: -1.6138606, -0.2685580, -1.6009877, -0.2947612, -1.0016675, 1.0169072
9: -8.8352432, -6.9607797, -8.8338871, -6.9638643, -0.9859619, 0.9877868

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720459
time: 4.70 seconds

## Relational analysis of NS_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720446
time: 6.22 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -8.5300465, -6.5595274, -8.5331144, -6.5594893, -0.9353554, 0.9373777
1: -13.8848372, -11.7883224, -13.8873720, -11.7882128, -1.1565647, 1.1571441
2: 7.0851421, 8.8582916, 7.0845509, 8.8587027, -0.9605842, 0.9580417
3: -4.7159262, -3.1610751, -4.7162127, -3.1608448, -1.3612432, 1.3769608
4: -10.4491482, -8.5371599, -10.4544516, -8.5371523, -1.1518698, 1.1630971
5: -10.2426205, -8.5618305, -10.2430830, -8.5606108, -0.8951621, 0.8932917
6: -12.8458290, -10.2887554, -12.8463249, -10.2885723, -1.5530748, 1.5917964
7: -3.9638541, -2.4751754, -3.9639525, -2.4741449, -1.1194425, 1.1247010
8: -1.6356111, -0.2677956, -1.6358404, -0.2668772, -1.0423341, 1.0045009
9: -8.8379955, -6.9617810, -8.8380671, -6.9609532, -0.9959860, 0.9889877

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3725362
time: 4.99 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3728437
time: 14.40 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -8.5382633, -6.5515003, -8.5376759, -6.5594296, -0.9393830, 0.9499967
1: -13.8914318, -11.7813005, -13.8911448, -11.7880478, -1.1628580, 1.1691942
2: 7.0833154, 8.8612709, 7.0836711, 8.8593044, -0.9621658, 0.9619017
3: -4.7183919, -3.1589141, -4.7166376, -3.1605048, -1.3637414, 1.3805642
4: -10.4632454, -8.5244970, -10.4623299, -8.5371332, -1.1591315, 1.1726019
5: -10.2470779, -8.5582314, -10.2437706, -8.5587912, -0.9015112, 0.8961453
6: -12.8482971, -10.2862701, -12.8470974, -10.2882738, -1.5556445, 1.5939703
7: -3.9675965, -2.4717124, -3.9641011, -2.4726100, -1.1244097, 1.1267018
8: -1.6388745, -0.2651219, -1.6361990, -0.2654929, -1.0452509, 1.0067554
9: -8.8405571, -6.9596591, -8.8381729, -6.9597278, -1.0002551, 0.9924603

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3725376
time: 11.15 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3728438
time: 12.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 46.23 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3717384
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3717381
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720459
NS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720446
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3725362
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3728437
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3725376
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 46.23
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3728438

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -8.5258808, -6.5601673, -8.5229111, -6.5645790, -0.9265904, 0.9282637
1: -13.8820839, -11.7896919, -13.8792458, -11.7914963, -1.1473393, 1.1472716
2: 7.0877013, 8.8560905, 7.0904903, 8.8513222, -0.9504576, 0.9522390
3: -4.7129660, -3.1641459, -4.7111564, -3.1695971, -1.3551426, 1.3572421
4: -10.4459610, -8.5467453, -10.4318123, -8.5531816, -1.1435990, 1.1318893
5: -10.2377949, -8.5623779, -10.2334919, -8.5658197, -0.8846259, 0.8840768
6: -12.8432693, -10.3111324, -12.8169384, -10.3273640, -1.5588522, 1.5482345
7: -3.9622774, -2.4865966, -3.9485984, -2.4945509, -1.1105905, 1.1005077
8: -1.6132581, -0.2708640, -1.5977373, -0.2974353, -0.9992137, 1.0114832
9: -8.8350639, -6.9628315, -8.8313179, -6.9659829, -0.9819679, 0.9821572

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717383
time: 5.27 seconds

## Relational analysis of NS_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717384
time: 6.57 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -8.5340910, -6.5521431, -8.5229111, -6.5645790, -0.9346862, 0.9363375
1: -13.8886814, -11.7826967, -13.8792458, -11.7914963, -1.1539874, 1.1543965
2: 7.0858736, 8.8590841, 7.0904903, 8.8513222, -0.9522443, 0.9551907
3: -4.7154503, -3.1619883, -4.7111564, -3.1695971, -1.3568320, 1.3593497
4: -10.4600430, -8.5340853, -10.4318123, -8.5531816, -1.1572838, 1.1430075
5: -10.2422571, -8.5587711, -10.2334919, -8.5658197, -0.8891597, 0.8875494
6: -12.8457584, -10.3086386, -12.8169384, -10.3273640, -1.5610247, 1.5505409
7: -3.9660099, -2.4831278, -3.9485984, -2.4945509, -1.1140666, 1.1038070
8: -1.6165128, -0.2681928, -1.5977373, -0.2974353, -1.0023227, 1.0140436
9: -8.8376293, -6.9607153, -8.8313179, -6.9659829, -0.9842439, 0.9843268

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 65

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 65

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717398
time: 4.06 seconds

## Relational analysis of NS_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717398
time: 7.01 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -8.5258808, -6.5601673, -8.5311069, -6.5565615, -0.9346535, 0.9363666
1: -13.8820839, -11.7896919, -13.8858433, -11.7845058, -1.1544700, 1.1539183
2: 7.0877013, 8.8560905, 7.0886574, 8.8543310, -0.9534197, 0.9540231
3: -4.7129660, -3.1641459, -4.7136364, -3.1674900, -1.3571959, 1.3589373
4: -10.4459610, -8.5467453, -10.4459009, -8.5405216, -1.1546261, 1.1455564
5: -10.2377949, -8.5623779, -10.2379408, -8.5622101, -0.8880997, 0.8885984
6: -12.8432693, -10.3111324, -12.8194294, -10.3248777, -1.5611429, 1.5504098
7: -3.9622774, -2.4865966, -3.9522784, -2.4910858, -1.1139002, 1.1039400
8: -1.6132581, -0.2708640, -1.6009877, -0.2947612, -1.0017786, 1.0145831
9: -8.8350639, -6.9628315, -8.8338871, -6.9638643, -0.9841399, 0.9844213

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 65

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 65

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B1_B2_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3720459
time: 3.61 seconds

## Relational analysis of NS_B1_B2_A1_A2

### Relational analysis result of NS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3720444
time: 6.37 seconds

## BFS NS instance: NS_B1_B2_A2

### Backsubstitution after applying NS history:
0: -8.5340910, -6.5521431, -8.5311069, -6.5565615, -0.9326622, 0.9343536
1: -13.8886814, -11.7826967, -13.8858433, -11.7845058, -1.1553464, 1.1552715
2: 7.0858736, 8.8590841, 7.0886574, 8.8543310, -0.9532149, 0.9549825
3: -4.7154503, -3.1619883, -4.7136364, -3.1674900, -1.3606992, 1.3628612
4: -10.4600430, -8.5340853, -10.4459009, -8.5405216, -1.1541314, 1.1424026
5: -10.2422571, -8.5587711, -10.2379408, -8.5622101, -0.8893929, 0.8888302
6: -12.8457584, -10.3086386, -12.8194294, -10.3248777, -1.5619493, 1.5513477
7: -3.9660099, -2.4831278, -3.9522784, -2.4910858, -1.1136441, 1.1035480
8: -1.6165128, -0.2681928, -1.6009877, -0.2947612, -1.0034027, 1.0156579
9: -8.8376293, -6.9607153, -8.8338871, -6.9638643, -0.9861374, 0.9863131

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B1_B2_A2_A1

### Relational analysis result of NS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3717398
time: 3.67 seconds

## Relational analysis of NS_B1_B2_A2_A2

### Relational analysis result of NS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3717398
time: 4.11 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.5300465, -6.5595274, -8.5300446, -6.5595279, -0.9353235, 0.9343035
1: -13.8848372, -11.7883224, -13.8848343, -11.7883253, -1.1555557, 1.1538153
2: 7.0851421, 8.8582916, 7.0851412, 8.8582916, -0.9601645, 0.9574409
3: -4.7159262, -3.1610751, -4.7159257, -3.1610785, -1.3606882, 1.3763514
4: -10.4491482, -8.5371599, -10.4491482, -8.5371666, -1.1518569, 1.1582322
5: -10.2426205, -8.5618305, -10.2426186, -8.5618334, -0.8939314, 0.8928068
6: -12.8458290, -10.2887554, -12.8458261, -10.2887707, -1.5528107, 1.5912347
7: -3.9638541, -2.4751754, -3.9638546, -2.4751818, -1.1181588, 1.1245623
8: -1.6356111, -0.2677956, -1.6355948, -0.2677960, -1.0416985, 1.0042965
9: -8.8379955, -6.9617810, -8.8379946, -6.9617801, -0.9946303, 0.9884970

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 65

## Relational analysis of NS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 65

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3725399
time: 3.64 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3717387
time: 5.48 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.5300465, -6.5595274, -8.5382595, -6.5515032, -0.9434013, 0.9423864
1: -13.8848372, -11.7883224, -13.8914318, -11.7812996, -1.1626840, 1.1604605
2: 7.0851421, 8.8582916, 7.0833158, 8.8612700, -0.9631028, 0.9592314
3: -4.7159262, -3.1610751, -4.7183905, -3.1589155, -1.3628297, 1.3780260
4: -10.4491482, -8.5371599, -10.4632463, -8.5245075, -1.1573734, 1.1648300
5: -10.2426205, -8.5618305, -10.2470722, -8.5582294, -0.8973999, 0.8973343
6: -12.8458290, -10.2887554, -12.8482962, -10.2862806, -1.5551138, 1.5928049
7: -3.9638541, -2.4751754, -3.9675968, -2.4717207, -1.1214542, 1.1250875
8: -1.6356111, -0.2677956, -1.6388583, -0.2651234, -1.0426993, 1.0073962
9: -8.8379955, -6.9617810, -8.8405581, -6.9596572, -0.9968047, 0.9907660

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 65

## Relational analysis of NS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 65

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3728436
time: 3.64 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3720426
time: 6.29 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.5382633, -6.5515003, -8.5300446, -6.5595279, -0.9434340, 0.9423752
1: -13.8914318, -11.7813005, -13.8848343, -11.7883253, -1.1621995, 1.1609426
2: 7.0833154, 8.8612709, 7.0851412, 8.8582916, -0.9619436, 0.9604113
3: -4.7183919, -3.1589141, -4.7159257, -3.1610785, -1.3623624, 1.3784666
4: -10.4632454, -8.5244970, -10.4491482, -8.5371666, -1.1627326, 1.1594825
5: -10.2470779, -8.5582314, -10.2426186, -8.5618334, -0.8984585, 0.8962750
6: -12.8482971, -10.2862701, -12.8458261, -10.2887707, -1.5549717, 1.5921297
7: -3.9675965, -2.4717124, -3.9638546, -2.4751818, -1.1212263, 1.1261067
8: -1.6388745, -0.2651219, -1.6355948, -0.2677960, -1.0429206, 1.0068588
9: -8.8405571, -6.9596591, -8.8379946, -6.9617801, -0.9969001, 0.9906714

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 65

## Relational analysis of NS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 65

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720428, upper bound: 0.3725375
time: 3.41 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720428, upper bound: 0.3717379
time: 5.25 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.5382633, -6.5515003, -8.5382595, -6.5515032, -0.9414246, 0.9403713
1: -13.8914318, -11.7813005, -13.8914318, -11.7812996, -1.1635499, 1.1618090
2: 7.0833154, 8.8612709, 7.0833158, 8.8612700, -0.9628897, 0.9602103
3: -4.7183919, -3.1589141, -4.7183905, -3.1589155, -1.3663201, 1.3815002
4: -10.4632454, -8.5244970, -10.4632463, -8.5245075, -1.1624227, 1.1687779
5: -10.2470779, -8.5582314, -10.2470722, -8.5582294, -0.8986864, 0.8975627
6: -12.8482971, -10.2862701, -12.8482962, -10.2862806, -1.5559349, 1.5947599
7: -3.9675965, -2.4717124, -3.9675968, -2.4717207, -1.1212072, 1.1270027
8: -1.6388745, -0.2651219, -1.6388583, -0.2651234, -1.0454764, 1.0084732
9: -8.8405571, -6.9596591, -8.8405581, -6.9596572, -0.9987626, 0.9926283

Time for backsubstitution: 22.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 65

## Relational analysis of NS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 65

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720429, upper bound: 0.3725375
time: 3.61 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720429, upper bound: 0.3717379
time: 5.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 37.25 seconds
NS_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717383
NS_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717384
NS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717398
NS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717391, upper bound: 0.3717398
NS_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3720459
NS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3720444
NS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3717398
NS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3717398
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3725399
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3717387
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3728436
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3717368, upper bound: 0.3720426
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3720428, upper bound: 0.3725375
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3720428, upper bound: 0.3717379
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3720429, upper bound: 0.3725375
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 2, lower bound: -0.3720429, upper bound: 0.3717379

## BFS NS instance: NS_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -8.5229111, -6.5645790, -8.5229111, -6.5645790, -0.9231248, 0.9231253
1: -13.8792458, -11.7914963, -13.8792458, -11.7914963, -1.1438732, 1.1438727
2: 7.0904903, 8.8513222, 7.0904903, 8.8513222, -0.9475002, 0.9475002
3: -4.7111564, -3.1695971, -4.7111564, -3.1695971, -1.3461995, 1.3461995
4: -10.4318123, -8.5531816, -10.4318123, -8.5531816, -1.1257710, 1.1257710
5: -10.2334919, -8.5658197, -10.2334919, -8.5658197, -0.8805804, 0.8805802
6: -12.8169384, -10.3273640, -12.8169384, -10.3273640, -1.5322456, 1.5322456
7: -3.9485984, -2.4945509, -3.9485984, -2.4945509, -1.0931568, 1.0931568
8: -1.5977373, -0.2974353, -1.5977373, -0.2974353, -0.9850850, 0.9850852
9: -8.8313179, -6.9659829, -8.8313179, -6.9659829, -0.9769206, 0.9769206

Time for backsubstitution: 21.90 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.43 + 547.56 = 606.99 seconds
