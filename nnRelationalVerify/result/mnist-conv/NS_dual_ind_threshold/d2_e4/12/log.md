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
execution time: IAR + RelationalAnalysis = 23.33 + 35.76 = 59.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3728446, upper bound: 0.3728441

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720451, upper bound: 0.3728441
time: 5.05 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3728441
time: 5.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.97 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.97
Output dim: 2, lower bound: -0.3720451, upper bound: 0.3728441
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.97
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3728441

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.5305357, -6.5644798, -8.5335140, -6.5600677, -0.9359827, 0.9343023
1: -13.8855553, -11.7912292, -13.8883953, -11.7894220, -1.1580248, 1.1580949
2: 7.0890179, 8.8523531, 7.0862308, 8.8571110, -0.9547741, 0.9529979
3: -4.7118793, -3.1690207, -4.7136874, -3.1635637, -1.3601408, 1.3580136
4: -10.4450283, -8.5531473, -10.4591522, -8.5467110, -1.1450353, 1.1567507
5: -10.2346210, -8.5627766, -10.2389441, -8.5593319, -0.8883066, 0.8888726
6: -12.8182182, -10.3268747, -12.8445501, -10.3106327, -1.5507336, 1.5613427
7: -3.9488356, -2.4919736, -3.9625266, -2.4840207, -1.1041217, 1.1141911
8: -1.5983412, -0.2951255, -1.6138616, -0.2685580, -1.0143070, 1.0020387
9: -8.8314934, -6.9639268, -8.8352432, -6.9607792, -0.9867563, 0.9865787

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 65

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720448, upper bound: 0.3725377
time: 4.72 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3720448, upper bound: 0.3728423
time: 5.27 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.5376806, -6.5594273, -8.5376835, -6.5594292, -0.9420049, 0.9430449
1: -13.8911495, -11.7880440, -13.8911476, -11.7880459, -1.1645718, 1.1663070
2: 7.0836720, 8.8593054, 7.0836678, 8.8593054, -0.9599943, 0.9626937
3: -4.7166386, -3.1605043, -4.7166386, -3.1605024, -1.3792753, 1.3635774
4: -10.4623375, -8.5371332, -10.4623394, -8.5371256, -1.1713595, 1.1650448
5: -10.2437716, -8.5587883, -10.2437725, -8.5587864, -0.8970568, 0.8981814
6: -12.8470974, -10.2882748, -12.8471031, -10.2882595, -1.5937362, 1.5553212
7: -3.9640999, -2.4726095, -3.9641006, -2.4726043, -1.1281819, 1.1217756
8: -1.6361995, -0.2654905, -1.6362171, -0.2654886, -1.0071268, 1.0445225
9: -8.8381748, -6.9597273, -8.8381767, -6.9597254, -0.9930844, 0.9992185

Time for backsubstitution: 22.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3720430
time: 6.02 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3728426
time: 5.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 34.71 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 2, lower bound: -0.3720448, upper bound: 0.3725377
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 2, lower bound: -0.3720448, upper bound: 0.3728423
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3720430
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 2, lower bound: -0.3728430, upper bound: 0.3728426

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8.5259762, -6.5645385, -8.5258808, -6.5601673, -0.9313455, 0.9266224
1: -13.8817806, -11.7913837, -13.8820839, -11.7896919, -1.1505985, 1.1483502
2: 7.0898962, 8.8517361, 7.0877013, 8.8560905, -0.9528403, 0.9508827
3: -4.7114477, -3.1693649, -4.7129660, -3.1641459, -1.3578534, 1.3556900
4: -10.4371271, -8.5531693, -10.4459610, -8.5467453, -1.1371694, 1.1436129
5: -10.2339497, -8.5645952, -10.2377949, -8.5623779, -0.8845520, 0.8858566
6: -12.8174391, -10.3271675, -12.8432693, -10.3111324, -1.5489759, 1.5591125
7: -3.9486926, -2.4935117, -3.9622774, -2.4865966, -1.1006789, 1.1118717
8: -1.5979819, -0.2965150, -1.6132581, -0.2708640, -1.0116849, 1.0001493
9: -8.8313904, -6.9651523, -8.8350639, -6.9628315, -0.9826488, 0.9833331

Time for backsubstitution: 22.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3725377
time: 3.80 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3725377
time: 3.76 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.5305328, -6.5644808, -8.5340910, -6.5521431, -0.9439774, 0.9306352
1: -13.8855534, -11.7912273, -13.8886814, -11.7826967, -1.1626425, 1.1546440
2: 7.0890179, 8.8523531, 7.0858736, 8.8590841, -0.9566832, 0.9524798
3: -4.7118797, -3.1690207, -4.7154503, -3.1619883, -1.3619003, 1.3581882
4: -10.4450197, -8.5531473, -10.4600430, -8.5340853, -1.1561201, 1.1508408
5: -10.2346230, -8.5627766, -10.2422571, -8.5587711, -0.8873968, 0.8922124
6: -12.8182163, -10.3268738, -12.8457584, -10.3086386, -1.5523772, 1.5616612
7: -3.9488351, -2.4919751, -3.9660099, -2.4831278, -1.1032686, 1.1172390
8: -1.5983427, -0.2951269, -1.6165128, -0.2681928, -1.0139346, 1.0046489
9: -8.8314934, -6.9639263, -8.8376293, -6.9607153, -0.9861374, 0.9876218

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3728438
time: 3.80 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3728437
time: 4.35 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.5376806, -6.5594273, -8.5305357, -6.5644798, -0.9378376, 0.9360497
1: -13.8911495, -11.7880440, -13.8855553, -11.7912292, -1.1607466, 1.1581569
2: 7.0836720, 8.8593054, 7.0890179, 8.8523531, -0.9553900, 0.9573429
3: -4.7166386, -3.1605043, -4.7118793, -3.1690207, -1.3536682, 1.3574662
4: -10.4623375, -8.5371332, -10.4450283, -8.5531473, -1.1553426, 1.1543777
5: -10.2437716, -8.5587883, -10.2346210, -8.5627766, -0.8941081, 0.8888800
6: -12.8470974, -10.2882748, -12.8182182, -10.3268747, -1.5542336, 1.5588269
7: -3.9640999, -2.4726095, -3.9488356, -2.4919736, -1.1090174, 1.1116395
8: -1.6361995, -0.2654905, -1.5983412, -0.2951255, -1.0101862, 1.0065045
9: -8.8381748, -6.9597273, -8.8314934, -6.9639268, -0.9880037, 0.9855309

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725364, upper bound: 0.3720424
time: 6.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728426, upper bound: 0.3720440
time: 5.78 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.5376806, -6.5594273, -8.5376806, -6.5594273, -0.9420033, 0.9420033
1: -13.8911495, -11.7880440, -13.8911495, -11.7880440, -1.1663041, 1.1663041
2: 7.0836720, 8.8593054, 7.0836720, 8.8593054, -0.9599926, 0.9599929
3: -4.7166386, -3.1605043, -4.7166386, -3.1605043, -1.3792677, 1.3792677
4: -10.4623375, -8.5371332, -10.4623375, -8.5371332, -1.1650400, 1.1650400
5: -10.2437716, -8.5587883, -10.2437716, -8.5587883, -0.8970568, 0.8970568
6: -12.8470974, -10.2882748, -12.8470974, -10.2882748, -1.5553198, 1.5553193
7: -3.9640999, -2.4726095, -3.9640999, -2.4726095, -1.1217713, 1.1217713
8: -1.6361995, -0.2654905, -1.6361995, -0.2654905, -1.0071259, 1.0071259
9: -8.8381748, -6.9597273, -8.8381748, -6.9597273, -0.9992151, 0.9992149

Time for backsubstitution: 22.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
time: 11.46 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3720440
time: 5.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 40.01 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3725377
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3725377
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3728438
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3728437
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3725364, upper bound: 0.3720424
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3728426, upper bound: 0.3720440
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 40.01
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3720440

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.5229111, -6.5645790, -8.5258808, -6.5601673, -0.9282637, 0.9265904
1: -13.8792458, -11.7914963, -13.8820839, -11.7896919, -1.1472721, 1.1473393
2: 7.0904903, 8.8513222, 7.0877013, 8.8560905, -0.9522390, 0.9504576
3: -4.7111564, -3.1695971, -4.7129660, -3.1641459, -1.3572421, 1.3551426
4: -10.4318123, -8.5531816, -10.4459610, -8.5467453, -1.1318893, 1.1435990
5: -10.2334919, -8.5658197, -10.2377949, -8.5623779, -0.8840771, 0.8846259
6: -12.8169384, -10.3273640, -12.8432693, -10.3111324, -1.5482349, 1.5588522
7: -3.9485984, -2.4945509, -3.9622774, -2.4865966, -1.1005077, 1.1105905
8: -1.5977373, -0.2974353, -1.6132581, -0.2708640, -1.0114832, 0.9992135
9: -8.8313179, -6.9659829, -8.8350639, -6.9628315, -0.9821572, 0.9819679

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3717363
time: 4.88 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3725361
time: 5.54 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.5311069, -6.5565615, -8.5258808, -6.5601673, -0.9363666, 0.9346538
1: -13.8858433, -11.7845058, -13.8820839, -11.7896919, -1.1539183, 1.1544700
2: 7.0886574, 8.8543310, 7.0877013, 8.8560905, -0.9540229, 0.9534197
3: -4.7136364, -3.1674900, -4.7129660, -3.1641459, -1.3589373, 1.3571959
4: -10.4459009, -8.5405216, -10.4459610, -8.5467453, -1.1455564, 1.1546261
5: -10.2379408, -8.5622101, -10.2377949, -8.5623779, -0.8885984, 0.8880999
6: -12.8194294, -10.3248777, -12.8432693, -10.3111324, -1.5504103, 1.5611424
7: -3.9522784, -2.4910858, -3.9622774, -2.4865966, -1.1039405, 1.1139002
8: -1.6009877, -0.2947612, -1.6132581, -0.2708640, -1.0145831, 1.0017786
9: -8.8338871, -6.9638643, -8.8350639, -6.9628315, -0.9844213, 0.9841399

Time for backsubstitution: 22.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 65

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3717378
time: 4.49 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3725376
time: 5.28 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.5229111, -6.5645790, -8.5340910, -6.5521431, -0.9363375, 0.9346859
1: -13.8792458, -11.7914963, -13.8886814, -11.7826967, -1.1543965, 1.1539874
2: 7.0904903, 8.8513222, 7.0858736, 8.8590841, -0.9551911, 0.9522440
3: -4.7111564, -3.1695971, -4.7154503, -3.1619883, -1.3593497, 1.3568320
4: -10.4318123, -8.5531816, -10.4600430, -8.5340853, -1.1430075, 1.1572838
5: -10.2334919, -8.5658197, -10.2422571, -8.5587711, -0.8875494, 0.8891594
6: -12.8169384, -10.3273640, -12.8457584, -10.3086386, -1.5505404, 1.5610247
7: -3.9485984, -2.4945509, -3.9660099, -2.4831278, -1.1038070, 1.1140666
8: -1.5977373, -0.2974353, -1.6165128, -0.2681928, -1.0140433, 1.0023227
9: -8.8313179, -6.9659829, -8.8376293, -6.9607153, -0.9843268, 0.9842441

Time for backsubstitution: 22.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 65

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3720439
time: 4.18 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3728437
time: 3.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.5311069, -6.5565615, -8.5340910, -6.5521431, -0.9343538, 0.9326622
1: -13.8858433, -11.7845058, -13.8886814, -11.7826967, -1.1552715, 1.1553464
2: 7.0886574, 8.8543310, 7.0858736, 8.8590841, -0.9549828, 0.9532149
3: -4.7136364, -3.1674900, -4.7154503, -3.1619883, -1.3628616, 1.3606992
4: -10.4459009, -8.5405216, -10.4600430, -8.5340853, -1.1424026, 1.1541312
5: -10.2379408, -8.5622101, -10.2422571, -8.5587711, -0.8888302, 0.8893929
6: -12.8194294, -10.3248777, -12.8457584, -10.3086386, -1.5513477, 1.5619483
7: -3.9522784, -2.4910858, -3.9660099, -2.4831278, -1.1035480, 1.1136441
8: -1.6009877, -0.2947612, -1.6165128, -0.2681928, -1.0156584, 1.0034029
9: -8.8338871, -6.9638643, -8.8376293, -6.9607153, -0.9863129, 0.9861374

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 65

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3717378
time: 3.91 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3725377
time: 3.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.5300446, -6.5595279, -8.5259762, -6.5645385, -0.9301481, 0.9314110
1: -13.8848343, -11.7883253, -13.8817806, -11.7913837, -1.1509962, 1.1507268
2: 7.0851412, 8.8582916, 7.0898962, 8.8517361, -0.9532731, 0.9554148
3: -4.7159257, -3.1610785, -4.7114477, -3.1693649, -1.3513498, 1.3551831
4: -10.4491482, -8.5371666, -10.4371271, -8.5531693, -1.1422281, 1.1449060
5: -10.2426186, -8.5618334, -10.2339497, -8.5645952, -0.8910875, 0.8851278
6: -12.8458261, -10.2887707, -12.8174391, -10.3271675, -1.5519996, 1.5568938
7: -3.9638546, -2.4751818, -3.9486926, -2.4935117, -1.1062555, 1.1081538
8: -1.6355948, -0.2677960, -1.5979819, -0.2965150, -1.0079937, 1.0038815
9: -8.8379946, -6.9617801, -8.8313904, -6.9651523, -0.9847589, 0.9814339

Time for backsubstitution: 22.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725364, upper bound: 0.3717387
time: 8.73 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 65

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.5382595, -6.5515032, -8.5305328, -6.5644808, -0.9341755, 0.9440467
1: -13.8914318, -11.7812996, -13.8855534, -11.7912273, -1.1572914, 1.1627893
2: 7.0833158, 8.8612700, 7.0890179, 8.8523531, -0.9548619, 0.9592440
3: -4.7183905, -3.1589155, -4.7118797, -3.1690207, -1.3538556, 1.3592644
4: -10.4632463, -8.5245075, -10.4450197, -8.5531473, -1.1494699, 1.1544042
5: -10.2470722, -8.5582294, -10.2346230, -8.5627766, -0.8974376, 0.8879678
6: -12.8482962, -10.2862806, -12.8182163, -10.3268738, -1.5540853, 1.5590744
7: -3.9675968, -2.4717207, -3.9488351, -2.4919751, -1.1091065, 1.1101522
8: -1.6388583, -0.2651234, -1.5983427, -0.2951269, -1.0109053, 1.0061235
9: -8.8405581, -6.9596572, -8.8314934, -6.9639263, -0.9890404, 0.9849064

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728426, upper bound: 0.3717397
time: 5.44 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728426, upper bound: 0.3720459
time: 4.48 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.5300446, -6.5595279, -8.5331144, -6.5594893, -0.9343340, 0.9373767
1: -13.8848343, -11.7883253, -13.8873720, -11.7882128, -1.1565628, 1.1588807
2: 7.0851412, 8.8582916, 7.0845509, 8.8587027, -0.9578691, 0.9580402
3: -4.7159257, -3.1610785, -4.7162127, -3.1608448, -1.3769131, 1.3769536
4: -10.4491482, -8.5371666, -10.4544516, -8.5371523, -1.1518645, 1.1571493
5: -10.2426186, -8.5618334, -10.2430830, -8.5606108, -0.8940368, 0.8932912
6: -12.8458261, -10.2887707, -12.8463249, -10.2885723, -1.5530720, 1.5535560
7: -3.9638546, -2.4751818, -3.9639525, -2.4741449, -1.1194382, 1.1183262
8: -1.6355948, -0.2677960, -1.6358404, -0.2668772, -1.0052299, 1.0044999
9: -8.8379946, -6.9617801, -8.8380671, -6.9609532, -0.9959831, 0.9951177

Time for backsubstitution: 22.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3717378
time: 4.17 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
time: 4.34 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.5382595, -6.5515032, -8.5376759, -6.5594296, -0.9383340, 0.9499953
1: -13.8914318, -11.7812996, -13.8911448, -11.7880478, -1.1628551, 1.1709256
2: 7.0833158, 8.8612700, 7.0836711, 8.8593044, -0.9594774, 0.9619007
3: -4.7183905, -3.1589155, -4.7166376, -3.1605048, -1.3794308, 1.3810048
4: -10.4632463, -8.5245075, -10.4623299, -8.5371332, -1.1591258, 1.1705287
5: -10.2470722, -8.5582294, -10.2437706, -8.5587912, -0.9003863, 0.8961451
6: -12.8482962, -10.2862806, -12.8470974, -10.2882738, -1.5556445, 1.5569606
7: -3.9675968, -2.4717207, -3.9641011, -2.4726100, -1.1247945, 1.1209283
8: -1.6388583, -0.2651234, -1.6361990, -0.2654929, -1.0097179, 1.0067544
9: -8.8405581, -6.9596572, -8.8381729, -6.9597278, -1.0002508, 0.9985907

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 65

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717378
time: 3.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3720427
time: 5.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.40 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3717363
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3725361
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3717378
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717411, upper bound: 0.3725376
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3720439
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3728437
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3717378
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3717387, upper bound: 0.3725377
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3728426, upper bound: 0.3717397
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3728426, upper bound: 0.3720459
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3717378
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3725366, upper bound: 0.3720440
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3717378
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.3728427, upper bound: 0.3720427

## BFS NS instance: NS_A1_B1_A1_B1

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

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 65

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 65

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.5229111, -6.5645790, -8.5300446, -6.5595279, -0.9283292, 0.9301162
1: -13.8792458, -11.7914963, -13.8848343, -11.7883253, -1.1473999, 1.1499853
2: 7.0904903, 8.8513222, 7.0851412, 8.8582916, -0.9548135, 0.9528482
3: -4.7111564, -3.1695971, -4.7159257, -3.1610785, -1.3545723, 1.3508024
4: -10.4318123, -8.5531816, -10.4491482, -8.5371666, -1.1412315, 1.1422143
5: -10.2334919, -8.5658197, -10.2426186, -8.5618334, -0.8846531, 0.8898568
6: -12.8169384, -10.3273640, -12.8458261, -10.2887707, -1.5563374, 1.5517397
7: -3.9485984, -2.4945509, -3.9638546, -2.4751818, -1.1080136, 1.1054029
8: -1.5977373, -0.2974353, -1.6355948, -0.2677960, -1.0036831, 1.0073588
9: -8.8313179, -6.9659829, -8.8379946, -6.9617801, -0.9809422, 0.9833934

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 65

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 65

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.5311069, -6.5565615, -8.5229111, -6.5645790, -0.9312282, 0.9311886
1: -13.8858433, -11.7845058, -13.8792458, -11.7914963, -1.1505189, 1.1510034
2: 7.0886574, 8.8543310, 7.0904903, 8.8513222, -0.9492841, 0.9504619
3: -4.7136364, -3.1674900, -4.7111564, -3.1695971, -1.3478942, 1.3482523
4: -10.4459009, -8.5405216, -10.4318123, -8.5531816, -1.1394382, 1.1368780
5: -10.2379408, -8.5622101, -10.2334919, -8.5658197, -0.8851018, 0.8840542
6: -12.8194294, -10.3248777, -12.8169384, -10.3273640, -1.5344210, 1.5345359
7: -3.9522784, -2.4910858, -3.9485984, -2.4945509, -1.0965896, 1.0964665
8: -1.6009877, -0.2947612, -1.5977373, -0.2974353, -0.9881849, 0.9876502
9: -8.8338871, -6.9638643, -8.8313179, -6.9659829, -0.9791842, 0.9790928

Time for backsubstitution: 22.12 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.09 + 561.32 = 620.40 seconds
