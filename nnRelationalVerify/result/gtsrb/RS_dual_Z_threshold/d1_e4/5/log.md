## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 9.9371055474


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6889954, 31.6889954)
1: (-12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314)
2: (-11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7814941, 18.7814903)
3: (-17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5188141, 23.5188141)
4: (-19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4152756, 22.4152756)
5: (-15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1424713, 24.1424713)
6: (-31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8810463, 19.8810425)
7: (-21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3111877, 26.3111877)
8: (-23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5465698, 29.5465622)
9: (-13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7479706, 20.7479706)
10: (-13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6405106, 27.6405106)
11: (-10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6082954, 17.6082916)
12: (-23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4471130, 34.4471130)
13: (-25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0297546, 31.0297470)
14: (-26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5937500, 39.5937576)
15: (-10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6990509, 21.6990547)
16: (-20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2432404, 25.2432404)
17: (-23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835)
18: (-11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0257187, 27.0257111)
19: (-7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7189674, 14.7189693)
20: (-6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4445305, 15.4445305)
21: (-7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4414215, 18.4414215)
22: (-5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3680840, 18.3680878)
23: (-2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8121185, 15.8121147)
24: (-5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4689064, 14.4689102)
25: (-0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4996719, 15.4996719)
26: (-12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547)
27: (-9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7664948, 19.7664948)
28: (-4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6528015, 17.6528015)
29: (-3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3217545, 16.3217525)
30: (-10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7909660, 17.7909622)
31: (-6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8262520, 18.8262482)
32: (-26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6703110, 22.6703110)
33: (-43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.9142456, 28.9142456)
34: (-36.2045822, -6.0280871, -36.2045822, -6.0280871, -23.0099030, 23.0099068)
35: (-26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9686661, 24.9686661)
36: (-27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4258728, 31.4258652)
37: (-44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6157761, 28.6157684)
38: (-31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414)
39: (-48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8900146, 33.8900070)
40: (-44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9788589, 19.9788513)
41: (-30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3767014, 21.3767014)
42: (-19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4059563, 15.4059582)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.36 + 40.80 = 43.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -9.9470526, upper bound: 9.9470526

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1652

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9461525, upper bound: 9.9461525
time: 31.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9461525, upper bound: 9.9461525
time: 23.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 55.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 55.68
Output dim: 25, lower bound: -9.9461525, upper bound: 9.9461525
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 55.68
Output dim: 25, lower bound: -9.9461525, upper bound: 9.9461525

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6843872, 31.6840286
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7784348, 18.7783165
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5116119, 23.5108795
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4129562, 22.4128571
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1377106, 24.1373825
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8579941, 19.8608055
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3124847, 26.3126450
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5393982, 29.5388412
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7347641, 20.7330055
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6340103, 27.6331482
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6080036, 17.6080894
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4435120, 34.4438324
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0216980, 31.0193405
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5756226, 39.5729675
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6916733, 21.6907196
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2322006, 25.2308502
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0236130, 27.0237808
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7195377, 14.7191200
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4408073, 15.4412918
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4394798, 18.4381790
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3681984, 18.3681946
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8126602, 15.8124237
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4687920, 14.4688110
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5011292, 15.4992142
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7565727, 19.7579193
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6492767, 17.6494141
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3193092, 16.3194237
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7831154, 17.7841339
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8296127, 18.8286285
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6527176, 22.6549683
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.9031754, 28.9044724
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9853516, 22.9882164
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9677429, 24.9676819
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4239960, 31.4241638
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6028442, 28.6042023
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8917389, 33.8914566
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9550209, 19.9580803
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3547592, 21.3573952
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3994007, 15.4004478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1660

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9437613
time: 18.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9390663
time: 22.03 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6840210, 31.6843872
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7783127, 18.7784309
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5108795, 23.5116119
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4128571, 22.4129562
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1373749, 24.1377029
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8608017, 19.8579941
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3126526, 26.3124847
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5388489, 29.5393906
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7330093, 20.7347679
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6331482, 27.6340179
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6080875, 17.6080093
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4438324, 34.4435120
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0193481, 31.0216904
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5729828, 39.5756073
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6907120, 21.6916771
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2308502, 25.2321968
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0237808, 27.0236130
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7191219, 14.7195358
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4412956, 15.4408035
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4381828, 18.4394760
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3681908, 18.3682022
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8124237, 15.8126602
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4688072, 14.4687920
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4992142, 15.5011292
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7579155, 19.7565804
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6494141, 17.6492767
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3194237, 16.3193092
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7841301, 17.7831154
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8286285, 18.8296127
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6549683, 22.6527176
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.9044724, 28.9031754
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9882202, 22.9853554
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9676819, 24.9677429
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4241638, 31.4239960
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6042023, 28.6028519
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8914642, 33.8917465
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9580803, 19.9550209
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3573914, 21.3547592
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4004498, 15.3994026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1660

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9437613
time: 23.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9390663
time: 23.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 48.50 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 48.50
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9437613
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 48.50
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9390663
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 48.50
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9437613
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 48.50
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9390663

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6813354, 31.6795578
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7759552, 18.7743950
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5104294, 23.5096970
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4124832, 22.4121704
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1374512, 24.1371460
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8527908, 19.8568802
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3097534, 26.3083191
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5355530, 29.5327530
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7286987, 20.7234001
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6334610, 27.6334686
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6072426, 17.6067390
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4413605, 34.4422684
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0205078, 31.0168304
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5761871, 39.5712280
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6912689, 21.6907005
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2259521, 25.2209663
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0225296, 27.0227509
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7190495, 14.7184982
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4408913, 15.4404488
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4358215, 18.4327278
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3625641, 18.3645554
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8125534, 15.8121605
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4643936, 14.4652405
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5011330, 15.4991646
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7565689, 19.7575645
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6486816, 17.6488228
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3162766, 16.3183823
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7775536, 17.7806778
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8293495, 18.8277817
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6499023, 22.6528091
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8941956, 28.8981018
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9651260, 22.9748230
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9595261, 24.9624939
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4230957, 31.4235001
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5968018, 28.5999603
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8916626, 33.8913956
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9517632, 19.9558411
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3496017, 21.3537064
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3985939, 15.3997288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9425161
time: 20.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9378168, upper bound: 9.9437613
time: 24.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6799316, 31.6809845
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7745132, 18.7758369
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5104218, 23.5097046
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4122696, 22.4123840
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1374664, 24.1371307
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8540726, 19.8556099
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3081665, 26.3099136
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5332947, 29.5349884
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7251587, 20.7269325
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6343307, 27.6325912
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6066551, 17.6073303
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4419556, 34.4416656
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0191650, 31.0181503
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5738678, 39.5735474
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6916580, 21.6903152
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2223129, 25.2246094
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0225906, 27.0226974
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7189121, 14.7186356
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4399605, 15.4413795
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4340210, 18.4345284
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3645630, 18.3625526
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8124008, 15.8123131
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4652252, 14.4644089
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5010834, 15.4992142
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7562332, 19.7579117
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6486816, 17.6488190
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3182678, 16.3163929
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7796593, 17.7785683
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8287697, 18.8283691
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6505585, 22.6521606
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8968048, 28.8955002
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9719620, 22.9679871
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9625549, 24.9594727
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4233398, 31.4232635
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5986099, 28.5981522
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8916779, 33.8913727
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9527779, 19.9548225
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3510742, 21.3522339
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3986816, 15.3996391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9378168
time: 23.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9425161, upper bound: 9.9390663
time: 23.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6809998, 31.6799164
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7758408, 18.7745171
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5097046, 23.5104218
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4123840, 22.4122696
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1371307, 24.1374664
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8556137, 19.8540688
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3099060, 26.3081589
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5350037, 29.5333023
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7269287, 20.7251625
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6325912, 27.6343307
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6073265, 17.6066589
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4416656, 34.4419556
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0181580, 31.0191803
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5735474, 39.5738678
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6903152, 21.6916580
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2246094, 25.2223129
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0226974, 27.0225906
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7186337, 14.7189121
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4413795, 15.4399605
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4345322, 18.4340210
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3625488, 18.3645630
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8123169, 15.8123970
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4644089, 14.4652252
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4992142, 15.5010834
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7579117, 19.7562256
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6488190, 17.6486816
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3163910, 16.3182678
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7785683, 17.7796593
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8283730, 18.8287659
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6521606, 22.6505585
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8955078, 28.8968048
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9679871, 22.9719620
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9594727, 24.9625549
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4232635, 31.4233322
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5981522, 28.5986099
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8913727, 33.8916855
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9548225, 19.9527817
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3522339, 21.3510742
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3996391, 15.3986816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9425161
time: 21.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9378168, upper bound: 9.9437613
time: 21.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6795654, 31.6813507
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7743988, 18.7759514
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5096970, 23.5104294
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4121704, 22.4124832
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1371460, 24.1374512
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8568802, 19.8527946
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3083191, 26.3097534
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5327454, 29.5355377
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7234039, 20.7286949
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6334686, 27.6334610
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6067390, 17.6072464
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4422760, 34.4413605
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0168457, 31.0204926
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5712280, 39.5761871
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6907043, 21.6912727
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2209625, 25.2259560
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0227509, 27.0225372
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7184963, 14.7190495
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4404488, 15.4408913
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4327316, 18.4358215
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3645554, 18.3625603
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8121567, 15.8125496
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4652405, 14.4643936
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4991646, 15.5011330
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7575607, 19.7565727
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6488190, 17.6486816
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3183823, 16.3162785
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7806740, 17.7775497
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8277855, 18.8293533
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6528091, 22.6499100
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8981018, 28.8942032
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9748230, 22.9651260
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9624939, 24.9595261
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4235077, 31.4231033
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5999603, 28.5968018
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8914032, 33.8916626
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9558372, 19.9517632
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3537064, 21.3496017
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3997269, 15.3985939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9378168
time: 22.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9425161, upper bound: 9.9390663
time: 22.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 46.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9425161
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9378168, upper bound: 9.9437613
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9378168
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9425161, upper bound: 9.9390663
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9390663, upper bound: 9.9425161
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9378168, upper bound: 9.9437613
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9437613, upper bound: 9.9378168
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.75
Output dim: 25, lower bound: -9.9425161, upper bound: 9.9390663

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6443176, 31.6468964
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7804718, 18.7789993
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4974594, 23.4982147
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4175110, 22.4169312
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1304016, 24.1306992
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8172798, 19.8206291
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3571472, 26.3513031
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5421753, 29.5392761
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7153320, 20.7107887
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6306534, 27.6315079
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5361099, 17.5257416
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4290924, 34.4294128
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9994965, 30.9972382
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5633087, 39.5569305
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6487808, 21.6547966
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2306824, 25.2236748
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0119553, 27.0116653
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7185402, 14.7173615
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4498634, 15.4425316
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4332275, 18.4248352
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3767815, 18.3803940
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8237038, 15.8254967
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4686317, 14.4689445
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5123081, 15.5103874
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7230568, 19.7222900
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6426277, 17.6414223
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3226166, 16.3245316
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7589989, 17.7570686
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8294144, 18.8278160
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6418381, 22.6455688
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8047028, 28.8203888
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9437180, 22.9614258
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9336929, 24.9405899
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4306335, 31.4321747
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6653061, 28.6788406
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8566742, 33.8652649
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9579506, 19.9649734
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3562775, 21.3612976
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4063759, 15.4094200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9364796, upper bound: 9.9266284
time: 23.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9231101, upper bound: 9.9400016
time: 19.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6486816, 31.6425323
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7805557, 18.7789154
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4989395, 23.4967346
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4172440, 22.4171982
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1309967, 24.1300964
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8165474, 19.8213654
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3527374, 26.3557053
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5420685, 29.5393829
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7160797, 20.7100258
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6315002, 27.6306610
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5262451, 17.5356064
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4284973, 34.4300003
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0009003, 30.9958344
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5618896, 39.5583344
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6553650, 21.6482086
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2286606, 25.2256927
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0114365, 27.0121765
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7179146, 14.7179890
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4429741, 15.4494190
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4279327, 18.4301300
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3783989, 18.3787766
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8258858, 15.8233147
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4680977, 14.4694786
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5123539, 15.5103378
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7213020, 19.7240448
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6412849, 17.6427650
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3224297, 16.3247204
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7539406, 17.7621231
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8293839, 18.8278427
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6426620, 22.6447372
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8164825, 28.8086014
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9517288, 22.9534187
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9376221, 24.9366684
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4317780, 31.4310303
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6756744, 28.6684723
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8655243, 33.8564148
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9608955, 19.9620209
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3571930, 21.3603821
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4082832, 15.4075146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9352272, upper bound: 9.9278736
time: 20.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9218591, upper bound: 9.9412469
time: 26.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6428833, 31.6483307
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7790298, 18.7804337
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4974594, 23.4982147
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4172974, 22.4171448
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1304169, 24.1306839
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8185463, 19.8193626
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3555450, 26.3528976
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5399323, 29.5415115
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7117920, 20.7143211
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6315231, 27.6306381
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5355225, 17.5263290
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4296875, 34.4288101
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9981842, 30.9985580
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5609894, 39.5592499
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6491623, 21.6544113
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2270355, 25.2273178
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0120163, 27.0116043
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7184029, 14.7174988
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4489326, 15.4434624
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4314270, 18.4266357
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3787880, 18.3783913
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8235512, 15.8256493
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4694633, 14.4681129
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5122585, 15.5104370
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7227058, 19.7226372
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6426277, 17.6414223
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3246078, 16.3225441
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7611046, 17.7549591
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8288269, 18.8284035
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6424866, 22.6449203
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8072968, 28.8177872
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9505539, 22.9545898
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9367218, 24.9375687
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4308624, 31.4319458
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6671219, 28.6770248
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8567047, 33.8652344
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9589577, 19.9639587
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3577499, 21.3598251
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4064674, 15.4093304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9412469, upper bound: 9.9218591
time: 17.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9278736, upper bound: 9.9352272
time: 22.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6472473, 31.6439590
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7791138, 18.7803574
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4989395, 23.4967346
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4170303, 22.4174118
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1310272, 24.1300735
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8178139, 19.8200951
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3511353, 26.3572998
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5398254, 29.5416260
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7125397, 20.7135658
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6323700, 27.6297836
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5256577, 17.5361977
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4290924, 34.4293976
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9995880, 30.9971542
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5595703, 39.5606537
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6557541, 21.6478233
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2250214, 25.2293358
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0114975, 27.0121231
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7177773, 14.7181263
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4420433, 15.4503517
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4261322, 18.4319305
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3804054, 18.3767700
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8257332, 15.8234673
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4689255, 14.4686508
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5123081, 15.5103874
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7209511, 19.7243958
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6412849, 17.6427650
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3244171, 16.3227310
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7560539, 17.7600174
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8288040, 18.8284302
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6433182, 22.6440887
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8190918, 28.8060074
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9585648, 22.9465828
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9406509, 24.9336395
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4320068, 31.4308014
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6774826, 28.6666565
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8655396, 33.8563919
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9619179, 19.9610023
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3586655, 21.3589096
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4083748, 15.4074249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9400016, upper bound: 9.9231101
time: 26.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9266284, upper bound: 9.9364796
time: 21.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6439514, 31.6472549
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7803574, 18.7791138
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4967346, 23.4989395
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4174118, 22.4170303
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1300812, 24.1310196
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8201027, 19.8178177
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3572998, 26.3511429
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5416260, 29.5398254
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7135620, 20.7125435
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6297836, 27.6323776
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5361938, 17.5256577
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4293976, 34.4291000
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9971466, 30.9995880
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5606537, 39.5595703
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6478195, 21.6557541
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2293320, 25.2250214
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0121231, 27.0114975
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7181282, 14.7177753
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4503517, 15.4420433
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4319305, 18.4261322
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3767738, 18.3804016
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8234673, 15.8257332
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4686508, 14.4689255
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5103855, 15.5123062
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7243919, 19.7209511
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6427650, 17.6412849
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3227310, 16.3244171
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7600136, 17.7560501
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8284302, 18.8288002
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6440887, 22.6433182
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8059998, 28.8190918
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9465866, 22.9585648
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9336395, 24.9406509
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4308014, 31.4320145
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6666641, 28.6774902
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8563843, 33.8655472
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9610023, 19.9619141
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3589096, 21.3586655
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4074287, 15.4083710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9364796, upper bound: 9.9266284
time: 25.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9231101, upper bound: 9.9400016
time: 17.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6483459, 31.6428909
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7804337, 18.7790298
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4982147, 23.4974594
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4171448, 22.4172974
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1306915, 24.1304169
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8193550, 19.8185539
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3529053, 26.3555450
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5415192, 29.5399323
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7143250, 20.7117882
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6306305, 27.6315231
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5263290, 17.5355225
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4288177, 34.4296875
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9985504, 30.9981842
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5592499, 39.5609741
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6544113, 21.6491661
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2273178, 25.2270355
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0116043, 27.0120163
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7174988, 14.7184029
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4434624, 15.4489307
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4266357, 18.4314270
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3783913, 18.3787842
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8256493, 15.8235512
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4681129, 14.4694633
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5104351, 15.5122604
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7226372, 19.7227097
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6414223, 17.6426277
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3225441, 16.3246059
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7549629, 17.7611084
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8284073, 18.8288231
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6449203, 22.6424866
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8177948, 28.8073044
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9545898, 22.9505577
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9375687, 24.9367218
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4319458, 31.4308701
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6770248, 28.6671219
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8652344, 33.8567047
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9639626, 19.9589615
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3598251, 21.3577499
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4093285, 15.4064674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9352272, upper bound: 9.9278736
time: 24.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9218591, upper bound: 9.9412469
time: 24.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6425171, 31.6486893
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7789154, 18.7805557
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4967346, 23.4989395
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4171982, 22.4172440
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1300964, 24.1310043
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8213692, 19.8165474
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3557129, 26.3527374
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5393829, 29.5420609
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7100372, 20.7160835
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6306610, 27.6315002
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5356064, 17.5262451
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4300079, 34.4284973
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9958344, 31.0009003
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5583344, 39.5618896
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6482086, 21.6553650
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2256927, 25.2286644
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0121765, 27.0114365
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7179909, 14.7179127
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4494209, 15.4429741
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4301300, 18.4279327
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3787727, 18.3783989
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8233147, 15.8258858
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4694786, 14.4680977
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5103397, 15.5123558
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7240486, 19.7212982
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6427650, 17.6412811
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3247223, 16.3224297
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7621269, 17.7539444
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8278427, 18.8293877
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6447372, 22.6426620
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8086090, 28.8164902
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9534225, 22.9517288
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9366684, 24.9376221
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4310303, 31.4317780
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6684723, 28.6756744
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8564148, 33.8655243
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9620247, 19.9608994
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3603821, 21.3571930
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4075127, 15.4082832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9412469, upper bound: 9.9218591
time: 18.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9278736, upper bound: 9.9352272
time: 20.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6469116, 31.6443253
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7789993, 18.7804718
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4982147, 23.4974594
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4169312, 22.4175110
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1306915, 24.1304016
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8206367, 19.8172798
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3513184, 26.3571320
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5392761, 29.5421753
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7107849, 20.7153282
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6315079, 27.6306534
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5257339, 17.5361137
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4294128, 34.4290924
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9972382, 30.9994965
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5569305, 39.5632935
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6547928, 21.6487808
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2236710, 25.2306824
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0116653, 27.0119553
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7173615, 14.7185402
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4425316, 15.4498634
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4248352, 18.4332275
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3803978, 18.3767815
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8254967, 15.8237038
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4689445, 14.4686317
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.5103855, 15.5123062
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7222862, 19.7230568
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6414223, 17.6426277
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3245316, 16.3226166
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7570686, 17.7589989
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8278122, 18.8294106
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6455688, 22.6418381
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8203888, 28.8047104
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9614258, 22.9437218
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9405899, 24.9336929
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4321747, 31.4306335
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6788406, 28.6653061
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8652649, 33.8566742
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9649696, 19.9579430
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3612976, 21.3562775
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4094200, 15.4063797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9400016, upper bound: 9.9231101
time: 18.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9266284, upper bound: 9.9364796
time: 22.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 43.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9364796, upper bound: 9.9266284
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9231101, upper bound: 9.9400016
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9352272, upper bound: 9.9278736
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9218591, upper bound: 9.9412469
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9412469, upper bound: 9.9218591
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9278736, upper bound: 9.9352272
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9400016, upper bound: 9.9231101
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9266284, upper bound: 9.9364796
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9364796, upper bound: 9.9266284
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9231101, upper bound: 9.9400016
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9352272, upper bound: 9.9278736
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9218591, upper bound: 9.9412469
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9412469, upper bound: 9.9218591
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9278736, upper bound: 9.9352272
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9400016, upper bound: 9.9231101
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 43.41
Output dim: 25, lower bound: -9.9266284, upper bound: 9.9364796

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6442871, 31.6467514
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7803116, 18.7788010
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4978333, 23.4978333
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4161224, 22.4147644
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1259308, 24.1274643
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8030586, 19.7994461
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3573914, 26.3507538
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5417023, 29.5384750
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7149124, 20.7087555
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6301804, 27.6300507
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5345840, 17.5245590
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4215698, 34.4184875
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9969940, 30.9937592
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5546265, 39.5506592
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6467361, 21.6551743
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2349167, 25.2217255
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0110626, 27.0122147
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7162724, 14.7154846
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4441338, 15.4383411
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4320564, 18.4239388
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3733063, 18.3816681
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8176384, 15.8212090
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4598503, 14.4628487
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4894867, 15.4945412
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7163010, 19.7176361
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6366730, 17.6372719
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3141556, 16.3207397
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7589073, 17.7570190
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8186646, 18.8205605
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6298676, 22.6275482
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7852478, 28.7931366
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9282608, 22.9390068
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9220047, 24.9238815
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4262543, 31.4259186
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6429062, 28.6457825
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8386230, 33.8390808
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9324379, 19.9263229
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3373795, 21.3333054
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3974953, 15.3962574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9137181, upper bound: 9.9395109
time: 22.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9226760, upper bound: 9.9309917
time: 18.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6486664, 31.6423798
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7803955, 18.7787170
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4993134, 23.4963531
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4158554, 22.4150314
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1265411, 24.1268616
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8023262, 19.8001823
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3529968, 26.3551559
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5415802, 29.5385818
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7156677, 20.7080002
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6310272, 27.6292038
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5247192, 17.5344238
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4209900, 34.4190750
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9983978, 30.9923553
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5532227, 39.5520630
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6533203, 21.6485825
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2328949, 25.2237396
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0105515, 27.0127335
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7156467, 14.7161140
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4372444, 15.4452286
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4267540, 18.4292374
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3749237, 18.3800468
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8198204, 15.8190269
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4593163, 14.4633865
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4895363, 15.4944916
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7145462, 19.7193947
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6353302, 17.6386147
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3139687, 16.3209267
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7538567, 17.7620735
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8186417, 18.8205833
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6306915, 22.6267166
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7970428, 28.7813492
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9362640, 22.9309998
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9259338, 24.9199600
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4273987, 31.4247742
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6532745, 28.6354141
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8474731, 33.8302383
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9353828, 19.9233704
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3382950, 21.3323898
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3993950, 15.3943520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9124788, upper bound: 9.9407560
time: 21.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9214267, upper bound: 9.9322322
time: 20.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6427460, 31.6483002
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7788315, 18.7802811
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4970856, 23.4985886
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4151306, 22.4157562
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1271820, 24.1262131
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.7973671, 19.8051453
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3549957, 26.3531647
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5391388, 29.5410385
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7097549, 20.7139053
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6300659, 27.6301575
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5343399, 17.5248032
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4187622, 34.4212952
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9947052, 30.9960480
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5547028, 39.5505829
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6495438, 21.6523628
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2250900, 25.2315483
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0125732, 27.0107117
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7165318, 14.7152328
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4447403, 15.4377327
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4305305, 18.4254608
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3800507, 18.3749161
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8192635, 15.8195839
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4633675, 14.4593315
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4964142, 15.4876175
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7180557, 19.7158813
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6384735, 17.6354713
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3208122, 16.3140831
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7610588, 17.7548752
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8215637, 18.8176575
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6244659, 22.6329498
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7800446, 28.7983322
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9281387, 22.9391289
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9200134, 24.9258804
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4246063, 31.4275665
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6340561, 28.6546249
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8305359, 33.8471832
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9203072, 19.9384460
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3297577, 21.3409309
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3933067, 15.4004440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9322322, upper bound: 9.9214267
time: 29.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9407560, upper bound: 9.9124788
time: 22.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6471100, 31.6439285
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7789154, 18.7801971
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4985580, 23.4971085
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4148636, 22.4160233
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1277924, 24.1256104
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.7966347, 19.8058815
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3505859, 26.3575592
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5390167, 29.5411530
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7105179, 20.7131500
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6309128, 27.6293106
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5244751, 17.5346680
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4181824, 34.4218903
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9961090, 30.9946442
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5532990, 39.5519867
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6561279, 21.6457748
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2230759, 25.2335663
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0120544, 27.0112305
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7158985, 14.7158604
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4378548, 15.4446220
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4252357, 18.4307556
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3816757, 18.3732948
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8214455, 15.8174019
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4628334, 14.4598694
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4964600, 15.4875679
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7163010, 19.7176361
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6371307, 17.6368141
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3206253, 16.3142700
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7560005, 17.7599297
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8215408, 18.8176804
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6252975, 22.6321182
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7918396, 28.7865448
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9361496, 22.9311218
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9239426, 24.9219513
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4257507, 31.4264221
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6444244, 28.6442566
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8393707, 33.8383408
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9232674, 19.9354935
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3306732, 21.3400154
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3952141, 15.3985405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9309917, upper bound: 9.9226760
time: 24.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9395109, upper bound: 9.9137181
time: 23.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6439362, 31.6471100
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7801971, 18.7789154
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4971085, 23.4985580
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4160233, 22.4148636
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1256104, 24.1277847
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8058815, 19.7966309
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3575592, 26.3505936
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5411530, 29.5390167
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7131500, 20.7105179
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6293106, 27.6309128
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5346680, 17.5244751
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4218903, 34.4181747
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9946442, 30.9961090
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5519867, 39.5532990
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6457748, 21.6561279
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2335663, 25.2230682
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0112305, 27.0120544
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7158604, 14.7159004
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4446220, 15.4378529
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4307594, 18.4252319
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3732986, 18.3816757
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8174019, 15.8214455
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4598694, 14.4628334
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4875679, 15.4964600
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7176361, 19.7163010
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6368103, 17.6371307
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3142700, 16.3206253
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7599297, 17.7560005
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8176804, 18.8215408
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6321182, 22.6252975
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7865448, 28.7918396
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9311218, 22.9361458
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9219513, 24.9239426
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4264221, 31.4257507
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6442566, 28.6444244
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8383484, 33.8393707
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9354897, 19.9232635
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3400192, 21.3306732
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3985405, 15.3952103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9137181, upper bound: 9.9395109
time: 22.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9226760, upper bound: 9.9309917
time: 22.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6483002, 31.6427383
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7802811, 18.7788315
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4985886, 23.4970779
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4157486, 22.4151306
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1262054, 24.1271820
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8051491, 19.7973671
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3531647, 26.3549957
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5410309, 29.5391312
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7139053, 20.7097549
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6301575, 27.6300659
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5248032, 17.5343399
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4212952, 34.4187622
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9960480, 30.9947052
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5505829, 39.5547028
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6523666, 21.6495438
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2315521, 25.2250862
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0107117, 27.0125656
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7152348, 14.7165298
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4377327, 15.4447403
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4254646, 18.4305305
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3749161, 18.3800545
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8195839, 15.8192635
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4593315, 14.4633675
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4876175, 15.4964142
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7158813, 19.7180557
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6354675, 17.6384773
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3140831, 16.3208122
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7548714, 17.7610550
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8176575, 18.8215675
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6329498, 22.6244659
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7983246, 28.7800522
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9391251, 22.9281387
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9258804, 24.9200134
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4275665, 31.4246063
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6546249, 28.6340637
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8471832, 33.8305283
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9384499, 19.9203110
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3409348, 21.3297577
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4004478, 15.3933067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9124788, upper bound: 9.9407560
time: 20.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9214267, upper bound: 9.9322322
time: 27.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6423798, 31.6486588
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7787170, 18.7803955
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4963531, 23.4993134
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4150314, 22.4158554
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1268616, 24.1265335
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8001747, 19.8023300
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3551636, 26.3530045
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5385895, 29.5415878
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7080002, 20.7156677
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6291962, 27.6310272
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5344238, 17.5247192
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4190826, 34.4209900
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9923553, 30.9983902
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5520630, 39.5532227
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6485825, 21.6533203
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2237396, 25.2328949
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0127335, 27.0105515
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7161121, 14.7156467
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4452286, 15.4372444
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4292336, 18.4267578
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3800507, 18.3749237
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8190269, 15.8198204
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4633865, 14.4593163
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4944916, 15.4895363
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7193909, 19.7145462
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6386108, 17.6353302
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3209267, 16.3139687
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7620735, 17.7538567
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8205795, 18.8186417
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6267166, 22.6306992
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7813568, 28.7970352
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9309998, 22.9362640
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9199600, 24.9259338
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4247742, 31.4273987
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6354141, 28.6532745
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8302460, 33.8474731
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9233742, 19.9353867
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3323898, 21.3382988
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3943520, 15.3993969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9322322, upper bound: 9.9214267
time: 26.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9407560, upper bound: 9.9124788
time: 24.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6467438, 31.6442947
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7788010, 18.7803116
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4978333, 23.4978333
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4147644, 22.4161224
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1274567, 24.1259308
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.7994423, 19.8030663
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3507538, 26.3573990
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5384674, 29.5417023
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7087555, 20.7149124
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6300507, 27.6301804
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5245590, 17.5345840
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4184875, 34.4215775
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9937592, 30.9969864
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5506592, 39.5546265
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6551743, 21.6467323
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2217255, 25.2349091
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0122223, 27.0110626
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7154865, 14.7162743
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4383392, 15.4441338
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4239388, 18.4320526
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3816681, 18.3733063
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8212090, 15.8176384
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4628487, 14.4598503
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4945412, 15.4894867
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7176361, 19.7163010
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6372681, 17.6366768
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3207397, 16.3141556
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7570152, 17.7589111
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8205566, 18.8186646
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6275482, 22.6298676
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7931366, 28.7852478
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9390106, 22.9282608
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9238815, 24.9220047
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4259186, 31.4262543
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6457825, 28.6429062
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8390808, 33.8386307
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9263191, 19.9324341
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3333054, 21.3373833
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3962593, 15.3974915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9309917, upper bound: 9.9226760
time: 21.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9395109, upper bound: 9.9137181
time: 19.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 43.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9137181, upper bound: 9.9395109
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9226760, upper bound: 9.9309917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9124788, upper bound: 9.9407560
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9214267, upper bound: 9.9322322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9322322, upper bound: 9.9214267
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9407560, upper bound: 9.9124788
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9309917, upper bound: 9.9226760
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9395109, upper bound: 9.9137181
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9137181, upper bound: 9.9395109
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9226760, upper bound: 9.9309917
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9124788, upper bound: 9.9407560
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9214267, upper bound: 9.9322322
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9322322, upper bound: 9.9214267
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9407560, upper bound: 9.9124788
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9309917, upper bound: 9.9226760
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.40
Output dim: 25, lower bound: -9.9395109, upper bound: 9.9137181

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6353760, 31.6366577
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7710114, 18.7682037
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4883499, 23.4877014
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4151382, 22.4137650
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1184235, 24.1194534
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8111801, 19.8077316
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3429108, 26.3344193
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5245972, 29.5191269
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6992188, 20.6910172
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6334000, 27.6335831
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5389805, 17.5282822
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4197388, 34.4167862
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0046692, 31.0007706
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5650177, 39.5597992
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6567383, 21.6660995
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2100830, 25.1936378
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0082703, 27.0095139
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7154770, 14.7146702
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4464149, 15.4401226
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4270210, 18.4176369
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3503838, 18.3615379
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8181152, 15.8216629
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4466248, 14.4507141
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4800873, 15.4853592
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7175751, 19.7185707
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6297264, 17.6308098
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2949524, 16.3038712
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7568588, 17.7569695
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8195877, 18.8214264
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6349869, 22.6327286
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7733612, 28.7822266
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8932419, 22.9076004
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9058304, 24.9094238
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4282379, 31.4279175
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6391602, 28.6422806
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8472290, 33.8472900
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9397163, 19.9336205
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3409157, 21.3368912
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4081440, 15.4062691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9119326, upper bound: 9.9277239
time: 22.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9059471, upper bound: 9.9380453
time: 24.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6397400, 31.6322861
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7711029, 18.7681198
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4898300, 23.4862213
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4148712, 22.4140320
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1190186, 24.1188431
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8104324, 19.8084679
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3385162, 26.3388214
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5244751, 29.5192337
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6999741, 20.6902618
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6342468, 27.6327362
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5291157, 17.5381508
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4191589, 34.4173737
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0060730, 30.9993668
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5636139, 39.5612030
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6633301, 21.6595116
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2080688, 25.1956558
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0077515, 27.0100250
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7148476, 14.7152977
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4395256, 15.4470100
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4217262, 18.4229355
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3520012, 18.3599167
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8202972, 15.8194809
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4460869, 14.4512520
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4801331, 15.4853096
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7158203, 19.7203255
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6283836, 17.6321526
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2947617, 16.3040600
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7518005, 17.7620277
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8195648, 18.8214493
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6358185, 22.6318970
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7851410, 28.7704468
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9012451, 22.8995972
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9097519, 24.9054947
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4293823, 31.4267731
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6495285, 28.6319199
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8560638, 33.8384399
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9426689, 19.9306679
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3418312, 21.3359756
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4100513, 15.4043636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9106918, upper bound: 9.9289699
time: 18.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9047073, upper bound: 9.9392953
time: 21.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6326599, 31.6393738
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7682343, 18.7709808
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4869537, 23.4891052
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4141312, 22.4147644
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1191711, 24.1187057
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8056564, 19.8132553
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3386536, 26.3386841
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5197754, 29.5239258
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6920242, 20.6982117
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6336060, 27.6333847
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5380650, 17.5291977
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4170685, 34.4194641
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0017090, 31.0037308
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5638428, 39.5609741
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6604691, 21.6623726
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.1969986, 25.2067184
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0098648, 27.0079193
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7157135, 14.7144337
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4465256, 15.4400139
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4242287, 18.4204292
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3599281, 18.3519936
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8197174, 15.8200607
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4512367, 14.4461060
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4872284, 15.4782143
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7189941, 19.7171669
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6320152, 17.6285248
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3039436, 16.2948799
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7610092, 17.7528229
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8224335, 18.8185806
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6296463, 22.6380692
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7691498, 28.7864380
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8967285, 22.9041138
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9055557, 24.9096985
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4266052, 31.4295425
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6305618, 28.6508789
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8387299, 33.8557816
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9276085, 19.9457283
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3333397, 21.3444672
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4033146, 15.4110985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9392953, upper bound: 9.9047073
time: 19.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9289699, upper bound: 9.9106918
time: 25.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6370239, 31.6350021
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7683105, 18.7708969
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4884262, 23.4876251
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4138641, 22.4150391
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1197815, 24.1181030
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8049240, 19.8139915
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3342590, 26.3430862
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5196838, 29.5240402
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6927795, 20.6974564
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6344452, 27.6325378
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5282001, 17.5390625
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4164734, 34.4200516
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0031128, 31.0023270
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5624390, 39.5623779
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6670532, 21.6557808
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.1949844, 25.2087326
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0093536, 27.0084381
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7150841, 14.7150612
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4396362, 15.4469032
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4189339, 18.4257240
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3615456, 18.3503723
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8218994, 15.8178787
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4507027, 14.4466400
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4872780, 15.4781647
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7172394, 19.7189217
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6306648, 17.6298676
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3037567, 16.2950668
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7559509, 17.7578773
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8224106, 18.8186073
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6304779, 22.6372375
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7809296, 28.7746582
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9047394, 22.8961067
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9094772, 24.9057693
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4277496, 31.4283981
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6409302, 28.6405106
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8475647, 33.8469391
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9305611, 19.9427757
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3342552, 21.3435516
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4052219, 15.4091930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9380453, upper bound: 9.9059471
time: 22.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9277239, upper bound: 9.9119326
time: 23.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6350098, 31.6370163
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7709045, 18.7683182
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4876251, 23.4884262
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4150391, 22.4138641
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1181030, 24.1197739
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8139877, 19.8049202
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3430939, 26.3342590
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5240479, 29.5196762
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6974564, 20.6927795
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6325378, 27.6344528
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5390644, 17.5282021
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4200439, 34.4164734
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0023193, 31.0031204
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5623779, 39.5624390
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6557846, 21.6670570
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2087326, 25.1949844
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0084381, 27.0093536
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7150612, 14.7150841
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4469032, 15.4396343
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4257240, 18.4189301
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3503685, 18.3615456
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8178787, 15.8218994
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4466400, 14.4506989
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4781647, 15.4872780
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7189178, 19.7172356
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6298714, 17.6306686
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2950668, 16.3037567
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7578735, 17.7559509
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8186035, 18.8224068
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6372452, 22.6304779
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7746582, 28.7809296
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8961029, 22.9047394
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9057693, 24.9094772
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4284058, 31.4277496
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6405106, 28.6409302
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8469391, 33.8475723
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9427757, 19.9305611
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3435478, 21.3342552
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4091930, 15.4052200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9119326, upper bound: 9.9277239
time: 21.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9059471, upper bound: 9.9380453
time: 20.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6393738, 31.6326447
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7709808, 18.7682343
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4891052, 23.4869461
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4147720, 22.4141312
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1187134, 24.1191635
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8132553, 19.8056526
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3386841, 26.3386536
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5239258, 29.5197830
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6982117, 20.6920242
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6333771, 27.6336060
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5291996, 17.5380669
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4194641, 34.4170685
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0037231, 31.0017166
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5609741, 39.5638428
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6623688, 21.6604691
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2067184, 25.1969986
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0079193, 27.0098648
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7144318, 14.7157135
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4400139, 15.4465218
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4204292, 18.4242287
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3519936, 18.3599243
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8200607, 15.8197174
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4461021, 14.4512367
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4782143, 15.4872284
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7171631, 19.7189903
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6285210, 17.6320114
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2948761, 16.3039455
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7528229, 17.7610092
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8185806, 18.8224335
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6380692, 22.6296463
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7864380, 28.7691422
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9041138, 22.8967323
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9096985, 24.9055557
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4295502, 31.4266052
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6508789, 28.6305618
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8557739, 33.8387299
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9457283, 19.9276085
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3444633, 21.3333435
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4110966, 15.4033165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9106918, upper bound: 9.9289699
time: 22.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9047073, upper bound: 9.9392953
time: 21.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6322937, 31.6397324
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7681122, 18.7710953
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4862213, 23.4898300
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4140320, 22.4148712
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1188354, 24.1190262
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8084641, 19.8104401
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3388214, 26.3385239
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5192261, 29.5244751
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6902618, 20.6999741
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6327362, 27.6342468
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5381489, 17.5291138
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4173737, 34.4191513
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9993744, 31.0060730
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5612030, 39.5636139
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6595154, 21.6633263
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.1956558, 25.2080650
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0100250, 27.0077515
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7152977, 14.7148476
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4470139, 15.4395256
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4229317, 18.4217224
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3599205, 18.3520012
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8194809, 15.8202972
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4512520, 14.4460907
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4853096, 15.4801331
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7203217, 19.7158279
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6321526, 17.6283836
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3040581, 16.2947655
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7620239, 17.7518044
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8214493, 18.8195648
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6318970, 22.6358185
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7704468, 28.7851410
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8995972, 22.9012489
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9054947, 24.9097519
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4267731, 31.4293747
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6319199, 28.6495285
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8384399, 33.8560638
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9306679, 19.9426689
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3359718, 21.3418350
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4043636, 15.4100494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9392953, upper bound: 9.9047073
time: 19.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9289699, upper bound: 9.9106918
time: 19.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6366577, 31.6353607
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7682037, 18.7710114
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4877014, 23.4883499
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4137650, 22.4151382
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1194458, 24.1184235
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8077316, 19.8111763
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3344269, 26.3429184
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5191345, 29.5245819
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6910172, 20.6992188
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6335831, 27.6334076
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5282841, 17.5389786
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4167938, 34.4197388
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0007782, 31.0046692
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5597992, 39.5650177
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6660995, 21.6567421
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.1936340, 25.2100830
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0095139, 27.0082703
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7146683, 14.7154770
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4401245, 15.4464149
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4176369, 18.4270210
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3615379, 18.3503799
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8216629, 15.8181152
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4507179, 14.4466248
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4853592, 15.4800873
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7185669, 19.7175827
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6308098, 17.6297264
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3038712, 16.2949524
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7569733, 17.7568588
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8214264, 18.8195877
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6327286, 22.6349945
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7822266, 28.7733612
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9076004, 22.8932419
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9094238, 24.9058304
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4279175, 31.4282379
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6422806, 28.6391602
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8472900, 33.8472290
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9336205, 19.9397163
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3368874, 21.3409195
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4062672, 15.4081459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9380453, upper bound: 9.9059471
time: 16.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9277239, upper bound: 9.9119326
time: 29.49 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 48.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9119326, upper bound: 9.9277239
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9059471, upper bound: 9.9380453
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9106918, upper bound: 9.9289699
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9047073, upper bound: 9.9392953
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9392953, upper bound: 9.9047073
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9289699, upper bound: 9.9106918
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9380453, upper bound: 9.9059471
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9277239, upper bound: 9.9119326
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9119326, upper bound: 9.9277239
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9059471, upper bound: 9.9380453
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9106918, upper bound: 9.9289699
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9047073, upper bound: 9.9392953
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9392953, upper bound: 9.9047073
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9289699, upper bound: 9.9106918
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9380453, upper bound: 9.9059471
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 48.43
Output dim: 25, lower bound: -9.9277239, upper bound: 9.9119326

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6352844, 31.6365662
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7712097, 18.7679291
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4883499, 23.4876404
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4141083, 22.4114532
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1165314, 24.1183548
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8086777, 19.8004570
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3434448, 26.3331985
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5243530, 29.5178223
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6997223, 20.6904640
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6330338, 27.6331406
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5383701, 17.5261345
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4180908, 34.4135284
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0042114, 30.9995346
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5628052, 39.5587769
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6532898, 21.6660423
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2116699, 25.1905022
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0081177, 27.0095215
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7154617, 14.7146454
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4454918, 15.4396362
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4266968, 18.4172287
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3465347, 18.3624535
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8153419, 15.8202591
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4444122, 14.4493179
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4714584, 15.4811325
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7166252, 19.7181816
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6275063, 17.6296806
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2893867, 16.3016758
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7566147, 17.7564125
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8161621, 18.8196640
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6319199, 22.6268463
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7698212, 28.7751770
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8911667, 22.9035606
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9039383, 24.9055710
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4269714, 31.4254684
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6330109, 28.6303024
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8417664, 33.8366318
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9323692, 19.9185257
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3360176, 21.3267975
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4066639, 15.4025764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9043696, upper bound: 9.9374069
time: 21.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9052536, upper bound: 9.9366170
time: 19.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6396484, 31.6321945
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7712860, 18.7678452
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4898300, 23.4861603
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4138336, 22.4117279
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1171265, 24.1177444
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8079453, 19.8011894
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3390350, 26.3376007
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5242462, 29.5179367
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7004700, 20.6897011
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6338806, 27.6322937
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5285053, 17.5360031
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4175110, 34.4141159
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0056152, 30.9981308
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5614014, 39.5601807
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6598816, 21.6594582
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2096558, 25.1925201
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0075989, 27.0100403
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7148323, 14.7152748
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4386024, 15.4465256
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4214020, 18.4225235
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3481522, 18.3608360
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8175240, 15.8180771
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4438744, 14.4498520
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4715080, 15.4810867
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7148705, 19.7199364
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6261635, 17.6310234
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2891960, 16.3018646
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7515564, 17.7614708
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8161392, 18.8196869
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6327515, 22.6260147
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7816010, 28.7633972
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8991776, 22.8955536
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9078674, 24.9016418
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4281311, 31.4243240
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6433716, 28.6199341
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8506012, 33.8277893
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9353218, 19.9155693
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3369331, 21.3258820
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4085674, 15.4006710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9031437, upper bound: 9.9386572
time: 23.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9040077, upper bound: 9.9378674
time: 26.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6325684, 31.6392975
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7679596, 18.7711716
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4868851, 23.4891052
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4118271, 22.4137344
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1180725, 24.1168060
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.7983780, 19.8107567
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3374329, 26.3392105
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5184784, 29.5236969
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6914673, 20.6987190
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6331558, 27.6330185
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5359211, 17.5285873
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4138031, 34.4178162
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0004730, 31.0032654
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5628204, 39.5587616
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6604156, 21.6589203
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.1938629, 25.2083054
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0098724, 27.0077667
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7156868, 14.7144203
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4460373, 15.4390907
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4238205, 18.4201050
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3608398, 18.3481483
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8183098, 15.8172874
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4498367, 14.4438896
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4830055, 15.4695854
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7185936, 19.7162094
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6308861, 17.6263008
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3017502, 16.2893143
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7604523, 17.7525787
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8206711, 18.8151550
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6237564, 22.6350021
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7621002, 28.7828979
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8926926, 22.9020386
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9017029, 24.9078064
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4241638, 31.4282913
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6185837, 28.6447296
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8280792, 33.8503113
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9125099, 19.9383850
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3232536, 21.3395653
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3996220, 15.4096146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9378675, upper bound: 9.9040077
time: 19.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9386572, upper bound: 9.9031437
time: 21.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6369324, 31.6349335
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7680511, 18.7710876
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4883652, 23.4876251
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4115601, 22.4140091
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1186676, 24.1162033
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.7976456, 19.8114891
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3330383, 26.3436050
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5183716, 29.5238037
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6922150, 20.6979561
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6340027, 27.6321716
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5260563, 17.5384521
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4132080, 34.4184113
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0018768, 31.0018616
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5614166, 39.5601654
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6669998, 21.6523323
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.1918488, 25.2103233
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0093613, 27.0082855
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7150612, 14.7150478
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4391518, 15.4459782
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4185257, 18.4254036
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3624649, 18.3465271
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8204918, 15.8151054
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4492989, 14.4444275
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4830513, 15.4695358
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7168388, 19.7179680
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6295433, 17.6276436
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3015594, 16.2895012
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7553940, 17.7576332
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8206482, 18.8151779
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6245880, 22.6341705
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7738800, 28.7711182
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9006958, 22.8940315
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9056244, 24.9038849
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4252930, 31.4271469
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6289520, 28.6343613
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8369141, 33.8414764
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9154625, 19.9354286
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3241692, 21.3386497
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4015293, 15.4077091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9366170, upper bound: 9.9052536
time: 21.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9374069, upper bound: 9.9043696
time: 22.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6349182, 31.6369247
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7710876, 18.7680435
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4876251, 23.4883652
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4140091, 22.4115524
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1161957, 24.1186752
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8114853, 19.7976418
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3436127, 26.3330383
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5238037, 29.5183716
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6979675, 20.6922188
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6321716, 27.6340027
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5384541, 17.5260544
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4184113, 34.4132080
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0018616, 31.0018845
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5601654, 39.5614166
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6523361, 21.6669998
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2103271, 25.1918488
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0082855, 27.0093613
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7150459, 14.7150612
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4459801, 15.4391479
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4253998, 18.4185257
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3465271, 18.3624611
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8151054, 15.8204956
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4444275, 14.4492989
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4695396, 15.4830551
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7179680, 19.7168427
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6276436, 17.6295433
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2895012, 16.3015614
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7576294, 17.7553940
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8151779, 18.8206482
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6341782, 22.6245880
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7711182, 28.7738800
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8940353, 22.9006996
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9038849, 24.9056244
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4271393, 31.4253006
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6343613, 28.6289520
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8414764, 33.8369141
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9354286, 19.9154663
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3386497, 21.3241653
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4077091, 15.4015274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9043696, upper bound: 9.9374069
time: 21.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9052536, upper bound: 9.9366170
time: 23.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6393127, 31.6325531
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7711639, 18.7679596
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4891052, 23.4868851
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4137344, 22.4118271
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1168060, 24.1180649
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8107529, 19.7983780
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3392181, 26.3374329
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5236969, 29.5184784
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6987152, 20.6914635
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6330109, 27.6331558
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5285892, 17.5359192
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4178162, 34.4138031
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0032654, 31.0004807
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5587616, 39.5628204
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6589203, 21.6604118
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2083054, 25.1938629
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0077667, 27.0098724
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7144203, 14.7156887
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4390907, 15.4460373
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4201050, 18.4238205
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3481445, 18.3608437
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8172874, 15.8183136
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4438896, 14.4498367
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4695854, 15.4830055
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7162132, 19.7185974
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6263008, 17.6308861
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2893105, 16.3017502
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7525787, 17.7604523
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8151550, 18.8206711
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6350021, 22.6237640
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7828979, 28.7620926
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9020386, 22.8926926
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9078064, 24.9017029
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4282837, 31.4241562
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6447296, 28.6185837
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8503113, 33.8280792
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9383812, 19.9125099
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3395653, 21.3232498
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4096127, 15.3996239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9031437, upper bound: 9.9386572
time: 23.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9040077, upper bound: 9.9378674
time: 25.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6322021, 31.6396637
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7678528, 18.7712860
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.4861603, 23.4898300
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4117279, 22.4138412
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1177521, 24.1171265
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8011856, 19.8079414
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3376007, 26.3390427
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5179291, 29.5242462
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6896973, 20.7004738
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6322937, 27.6338806
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5360050, 17.5285034
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4141083, 34.4175034
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9981384, 31.0056076
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5601807, 39.5614014
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6594543, 21.6598778
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.1925201, 25.2096558
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0100403, 27.0075989
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7152748, 14.7148342
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4465256, 15.4386024
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4225311, 18.4214020
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3608322, 18.3481560
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8180809, 15.8175240
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4498520, 14.4438744
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4810829, 15.4715042
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7199364, 19.7148705
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6310234, 17.6261597
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3018646, 16.2891998
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7614670, 17.7515602
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8196869, 18.8161354
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6260147, 22.6327515
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7633972, 28.7816010
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8955536, 22.8991737
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9016418, 24.9078674
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4243164, 31.4281235
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6199341, 28.6433716
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8277893, 33.8506012
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9155693, 19.9353256
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3258858, 21.3369293
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4006710, 15.4085655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9378675, upper bound: 9.9040077
time: 19.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9386572, upper bound: 9.9031437
time: 25.79 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 47.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9043696, upper bound: 9.9374069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9052536, upper bound: 9.9366170
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9031437, upper bound: 9.9386572
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9040077, upper bound: 9.9378674
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9378675, upper bound: 9.9040077
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9386572, upper bound: 9.9031437
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9366170, upper bound: 9.9052536
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9374069, upper bound: 9.9043696
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9043696, upper bound: 9.9374069
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9052536, upper bound: 9.9366170
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9031437, upper bound: 9.9386572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9040077, upper bound: 9.9378674
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9378675, upper bound: 9.9040077
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 47.07
Output dim: 25, lower bound: -9.9386572, upper bound: 9.9031437
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 47.07
Output dim: 25, lower bound: -9.9380453, upper bound: 9.9059471

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 43.16 + 1779.65 = 1822.81 seconds
