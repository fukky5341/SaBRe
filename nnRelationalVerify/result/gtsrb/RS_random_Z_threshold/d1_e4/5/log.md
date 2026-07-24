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
execution time: IAR + RelationalAnalysis = 2.41 + 40.84 = 43.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -9.9470526, upper bound: 9.9470526

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1412

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9445473, upper bound: 9.9311804
time: 26.08 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9311805, upper bound: 9.9445473
time: 22.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 48.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 48.47
Output dim: 25, lower bound: -9.9445473, upper bound: 9.9311804
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 48.47
Output dim: 25, lower bound: -9.9311805, upper bound: 9.9445473

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6888275, 31.6889496
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7812881, 18.7813263
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5184402, 23.5191879
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4131088, 22.4138870
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1392365, 24.1380005
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8598671, 19.8668365
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3106232, 26.3114471
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5457764, 29.5461197
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7459335, 20.7475471
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6390305, 27.6400223
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6071091, 17.6067619
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4361877, 34.4396057
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0262756, 31.0272446
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5874786, 39.5850906
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6994324, 21.6970062
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2412949, 25.2474747
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0262756, 27.0248260
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7170944, 14.7167053
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4403496, 15.4388084
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4405212, 18.4402428
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3693504, 18.3646049
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8078308, 15.8060493
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4628105, 14.4601212
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4838295, 15.4768562
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7618484, 19.7597466
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6486435, 17.6468468
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3179626, 16.3132973
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7908974, 17.7908592
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8189964, 18.8155060
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6523056, 22.6583481
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8869934, 28.8947983
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9874878, 22.9944458
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9519501, 24.9569702
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4196014, 31.4214706
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5827179, 28.5933762
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8638458, 33.8719711
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9402008, 19.9533386
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3487015, 21.3578033
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3928032, 15.3970757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1384

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9410548, upper bound: 9.9309859
time: 16.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9443527, upper bound: 9.9276871
time: 20.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6889496, 31.6888275
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7813263, 18.7812881
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5191879, 23.5184402
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4138870, 22.4131088
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1380005, 24.1392288
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8668327, 19.8598633
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3114471, 26.3106308
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5461121, 29.5457840
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7475510, 20.7459297
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6400146, 27.6390381
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6067581, 17.6071091
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4396057, 34.4361877
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0272522, 31.0262756
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5850983, 39.5874863
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6970062, 21.6994286
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2474747, 25.2412949
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0248260, 27.0262680
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7167053, 14.7170944
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4388084, 15.4403496
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4402466, 18.4405212
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3646049, 18.3693542
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8060455, 15.8078270
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4601212, 14.4628105
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4768562, 15.4838333
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7597427, 19.7618484
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6468506, 17.6486473
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3132973, 16.3179626
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7908592, 17.7908936
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8155022, 18.8189926
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6583481, 22.6522980
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8947906, 28.8870010
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9944458, 22.9874878
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9569702, 24.9519501
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4214630, 31.4195938
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5933762, 28.5827179
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8719788, 33.8638458
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9533386, 19.9402008
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3578033, 21.3487053
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3970757, 15.3927994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 972

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9189615, upper bound: 9.9436999
time: 20.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9302719, upper bound: 9.9317417
time: 26.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 49.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 49.80
Output dim: 25, lower bound: -9.9410548, upper bound: 9.9309859
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 49.80
Output dim: 25, lower bound: -9.9443527, upper bound: 9.9276871
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 49.80
Output dim: 25, lower bound: -9.9189615, upper bound: 9.9436999
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 49.80
Output dim: 25, lower bound: -9.9302719, upper bound: 9.9317417

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6557922, 31.6517258
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7792740, 18.7792053
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5199280, 23.5207748
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4186859, 22.4211349
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1360779, 24.1366577
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8403282, 19.8503761
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3050461, 26.3072739
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5458527, 29.5462265
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7217789, 20.7201080
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6354675, 27.6354523
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5992832, 17.5996628
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4382019, 34.4413071
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0153656, 31.0149231
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5843811, 39.5822296
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6921234, 21.6882515
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2336121, 25.2375870
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0222321, 27.0211487
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7160988, 14.7158241
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4150467, 15.4162445
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4362679, 18.4363213
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3579025, 18.3545303
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8139000, 15.8092613
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4549713, 14.4531746
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4759789, 15.4698715
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7477188, 19.7471695
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6484032, 17.6466103
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3055267, 16.3023262
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7605247, 17.7639008
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8057365, 18.8037987
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6520462, 22.6580811
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8555832, 28.8595581
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9802628, 22.9862251
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9364700, 24.9396286
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4159698, 31.4174194
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5253067, 28.5289154
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8374023, 33.8421249
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9162140, 19.9263306
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3258743, 21.3321762
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3747272, 15.3765507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1417

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1455

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9408740, upper bound: 9.9307981
time: 23.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9408675, upper bound: 9.9308046
time: 15.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6516113, 31.6559143
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7791748, 18.7793121
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5200272, 23.5206757
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4203568, 22.4194641
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1378784, 24.1348419
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8434029, 19.8472977
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3064499, 26.3058548
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5458832, 29.5461731
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7184906, 20.7233963
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6344604, 27.6364594
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6000080, 17.5989380
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4378967, 34.4416122
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0139618, 31.0163345
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5846252, 39.5819855
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6906738, 21.6896935
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2314072, 25.2397919
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0225906, 27.0207825
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7162132, 14.7157059
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4177856, 15.4135056
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4366035, 18.4359932
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3592758, 18.3531532
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8110390, 15.8121185
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4558601, 14.4522858
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4768448, 15.4690056
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7492752, 19.7456245
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6484108, 17.6466064
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3069954, 16.3008614
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7639427, 17.7604904
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8072853, 18.8022461
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6520309, 22.6580963
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8517685, 28.8633804
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9792709, 22.9872169
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9346085, 24.9414902
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4155426, 31.4178391
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5182571, 28.5359650
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8339996, 33.8455353
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9131927, 19.9293556
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3230743, 21.3349724
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3722782, 15.3790054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1652

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9178438, upper bound: 9.9005042
time: 22.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9171712, upper bound: 9.9011739
time: 19.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6889496, 31.6888275
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7812805, 18.7807159
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5191574, 23.5181961
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4131470, 22.4116974
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1379089, 24.1391754
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8690948, 19.8597565
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3114700, 26.3091888
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5455017, 29.5442886
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7474518, 20.7455406
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6398163, 27.6389389
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6078377, 17.6070366
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4394226, 34.4356842
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0271454, 31.0260391
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5848694, 39.5879364
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6969299, 21.7010078
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2487717, 25.2405319
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0247803, 27.0262604
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7167053, 14.7170944
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4386559, 15.4402885
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4402275, 18.4404945
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3628654, 18.3699188
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8049812, 15.8073463
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4593887, 14.4626846
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4751930, 15.4845734
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7597046, 19.7618370
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6455879, 17.6479645
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3103027, 16.3167229
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7906609, 17.7907715
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8144875, 18.8185577
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6596680, 22.6519775
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8946152, 28.8853455
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9940262, 22.9865723
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9567184, 24.9512024
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4211731, 31.4187164
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5921097, 28.5789337
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8704071, 33.8598404
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9535980, 19.9366570
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3591766, 21.3475113
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3980007, 15.3925991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1790

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 792

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9188837, upper bound: 9.9433883
time: 23.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9186499, upper bound: 9.9436220
time: 18.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 44.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 44.59
Output dim: 25, lower bound: -9.9408740, upper bound: 9.9307981
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 44.59
Output dim: 25, lower bound: -9.9408675, upper bound: 9.9308046
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 44.59
Output dim: 25, lower bound: -9.9178438, upper bound: 9.9005042
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 44.59
Output dim: 25, lower bound: -9.9171712, upper bound: 9.9011739
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 44.59
Output dim: 25, lower bound: -9.9188837, upper bound: 9.9433883
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 44.59
Output dim: 25, lower bound: -9.9186499, upper bound: 9.9436220

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6559753, 31.6518631
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7791672, 18.7791214
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5199966, 23.5208206
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4188690, 22.4214172
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1361389, 24.1366959
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8404312, 19.8504524
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3048630, 26.3071136
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5458679, 29.5462723
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7216415, 20.7199326
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6351929, 27.6351318
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5992584, 17.5996323
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4383240, 34.4413757
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0156403, 31.0152588
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5840912, 39.5818481
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6921158, 21.6882477
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2332153, 25.2372017
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0222626, 27.0211792
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7161751, 14.7159290
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4149399, 15.4161224
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4357109, 18.4356995
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3579483, 18.3545837
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8138390, 15.8091621
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4550362, 14.4532471
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4754620, 15.4693604
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7480545, 19.7474709
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6483955, 17.6465988
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3053818, 16.3021755
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7603035, 17.7636948
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8058205, 18.8039017
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6517944, 22.6577911
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8549957, 28.8590927
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9800186, 22.9860229
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9364395, 24.9396057
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4160309, 31.4174957
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5248795, 28.5285339
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8365784, 33.8414307
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9160805, 19.9262924
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3257713, 21.3320732
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3747940, 15.3765316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9394124, upper bound: 9.9230439
time: 21.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9290903, upper bound: 9.9290468
time: 27.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6559448, 31.6519089
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7791901, 18.7791061
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5199661, 23.5208511
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4189682, 22.4213181
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1361084, 24.1367264
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8404007, 19.8504753
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3048782, 26.3070984
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5458984, 29.5462570
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7216110, 20.7199707
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6351471, 27.6351700
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5992508, 17.5996399
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4382629, 34.4414291
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0157013, 31.0151978
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5839996, 39.5819397
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6921158, 21.6882439
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2332230, 25.2371902
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0222626, 27.0211792
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7162056, 14.7159042
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4149246, 15.4161415
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4356499, 18.4357643
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3579559, 18.3545761
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8138008, 15.8092041
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4550438, 14.4532394
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4754696, 15.4693527
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7480316, 19.7474976
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6483955, 17.6465988
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3053780, 16.3021812
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7603188, 17.7636833
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8058434, 18.8038826
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6517639, 22.6578293
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8551178, 28.8589783
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9800568, 22.9859810
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9364471, 24.9395981
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4160309, 31.4174957
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5249252, 28.5284882
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8367157, 33.8412933
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9161720, 19.9262047
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3257713, 21.3320770
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3747101, 15.3766136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1703

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1433

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9397638, upper bound: 9.9307629
time: 15.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9408249, upper bound: 9.9296984
time: 21.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6882477, 31.6893845
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7799606, 18.7787476
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5191193, 23.5182114
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4090958, 22.4059296
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1365204, 24.1371994
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8680038, 19.8616409
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3099213, 26.3069687
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5454559, 29.5440292
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7472534, 20.7459068
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6393509, 27.6384583
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6078777, 17.6058578
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4351654, 34.4327011
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0259552, 31.0244980
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5864716, 39.5877838
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6960678, 21.7001991
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2484360, 25.2402534
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0257645, 27.0258865
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7177620, 14.7166634
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4372139, 15.4380722
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4408455, 18.4391174
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3633423, 18.3695679
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8045197, 15.8070183
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4600716, 14.4623566
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4753990, 15.4838867
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7595558, 19.7617912
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6456566, 17.6479568
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3105698, 16.3166924
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7906723, 17.7905807
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8138199, 18.8160286
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6594543, 22.6543121
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8942642, 28.8889236
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9932098, 22.9894409
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9566498, 24.9519424
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4210968, 31.4192581
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5814896, 28.5721283
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8702545, 33.8609848
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9380951, 19.9259262
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3564529, 21.3479843
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3890667, 15.3880825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1609

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9188619, upper bound: 9.9432268
time: 14.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9187220, upper bound: 9.9433686
time: 22.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6894989, 31.6881256
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7793198, 18.7793961
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5191727, 23.5181580
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4073792, 22.4076462
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1359253, 24.1377869
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8709869, 19.8586617
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3092499, 26.3076248
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5452728, 29.5442200
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7478027, 20.7453499
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6393280, 27.6384735
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6066647, 17.6070786
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4364319, 34.4314346
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0256042, 31.0248413
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5847168, 39.5895233
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6961212, 21.7001495
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2484970, 25.2401962
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0244064, 27.0272446
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7162704, 14.7181549
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4364357, 15.4388466
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4388466, 18.4411087
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3625107, 18.3703918
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8046494, 15.8068848
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4590569, 14.4633713
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4745064, 15.4847832
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7596550, 19.7616959
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6455803, 17.6480331
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3102722, 16.3169918
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7904739, 17.7907791
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8119583, 18.8178864
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6620026, 22.6517639
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8981857, 28.8850021
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9968948, 22.9857559
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9574509, 24.9511414
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4217072, 31.4186478
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5853043, 28.5683212
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8715515, 33.8596802
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9428711, 19.9211502
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3596497, 21.3447838
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3934841, 15.3836594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1703

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9179231, upper bound: 9.9422049
time: 14.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9172239, upper bound: 9.9428997
time: 15.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 32.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9394124, upper bound: 9.9230439
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9290903, upper bound: 9.9290468
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9397638, upper bound: 9.9307629
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9408249, upper bound: 9.9296984
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9188619, upper bound: 9.9432268
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9187220, upper bound: 9.9433686
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9179231, upper bound: 9.9422049
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.63
Output dim: 25, lower bound: -9.9172239, upper bound: 9.9428997

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6558990, 31.6518097
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7789001, 18.7793121
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5199432, 23.5208282
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4165649, 22.4203873
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1350555, 24.1348038
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8331528, 19.8479538
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3036575, 26.3076401
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5445709, 29.5460434
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7210846, 20.7204437
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6347351, 27.6347504
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5971184, 17.5990295
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4350586, 34.4397278
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0144196, 31.0148163
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5830688, 39.5796509
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6920624, 21.6848068
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2300720, 25.2387924
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0222855, 27.0210419
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7161598, 14.7159195
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4144669, 15.4152069
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4353142, 18.4353867
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3588562, 18.3507309
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8124466, 15.8064003
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4536362, 14.4510345
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4712372, 15.4607353
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7476654, 19.7465210
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6472626, 17.6443710
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3031921, 16.2966137
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7597351, 17.7634430
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8040657, 18.8004761
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6459122, 22.6547241
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8479462, 28.8555527
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9759750, 22.9839478
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9325867, 24.9377136
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4135437, 31.4162064
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5129089, 28.5223999
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8259125, 33.8359451
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9009819, 19.9189453
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3156776, 21.3271637
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3711014, 15.3750496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1411

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1315

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9386853, upper bound: 9.9229875
time: 22.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9393568, upper bound: 9.9223153
time: 21.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6487274, 31.6457367
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7826767, 18.7824631
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5197983, 23.5206833
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4153137, 22.4177856
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1406860, 24.1416855
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8083725, 19.8151894
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3257446, 26.3262329
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5540161, 29.5538940
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7100601, 20.7085037
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6354599, 27.6360092
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5667839, 17.5632820
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4403534, 34.4434967
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0278931, 31.0289993
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5840302, 39.5819855
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6656723, 21.6642723
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2370529, 25.2405472
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0097275, 27.0083466
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7161026, 14.7158051
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.3897362, 15.3891335
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4380569, 18.4370155
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3716011, 18.3706512
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8135757, 15.8089981
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4550323, 14.4532776
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4806385, 15.4753494
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7280502, 19.7255287
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6488533, 17.6469994
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3094215, 16.3066864
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7700500, 17.7707939
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8055763, 18.8036652
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6453400, 22.6512604
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7996826, 28.8071213
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9763489, 22.9844704
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9019623, 24.9076614
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4169006, 31.4184341
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5292816, 28.5348969
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8351135, 33.8430481
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9161987, 19.9263000
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3226852, 21.3290215
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3707066, 15.3727989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1297

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1596

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9369691, upper bound: 9.9304602
time: 21.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9394633, upper bound: 9.9279815
time: 22.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6497650, 31.6447144
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7825470, 18.7826004
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5198059, 23.5206757
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4154434, 22.4176559
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1410675, 24.1413116
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8051224, 19.8184395
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3240051, 26.3279648
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5535278, 29.5543823
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7101440, 20.7084274
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6359940, 27.6354828
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5628929, 17.5671730
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4403381, 34.4435196
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0294952, 31.0273819
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5840302, 39.5819855
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6681442, 21.6618004
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2365799, 25.2410202
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0094299, 27.0086365
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7161026, 14.7158070
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.3879204, 15.3909492
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4369049, 18.4381714
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3740273, 18.3682251
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8135986, 15.8089790
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4550819, 14.4532318
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4814701, 15.4745255
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7260666, 19.7275124
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6487999, 17.6470528
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3098793, 16.3062286
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7674332, 17.7734108
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8056221, 18.8036194
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6451950, 22.6514053
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8032532, 28.8035507
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9785461, 22.9822731
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9045105, 24.9051132
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4169769, 31.4183655
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5313263, 28.5328522
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8384705, 33.8396988
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9162750, 19.9262352
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3227158, 21.3289948
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3708973, 15.3726082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1381

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 987

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9355296, upper bound: 9.9294148
time: 20.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9405411, upper bound: 9.9244050
time: 21.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6886597, 31.6897049
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7768250, 18.7763062
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5191956, 23.5181808
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4092026, 22.4061890
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1340408, 24.1348267
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8682442, 19.8615189
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3056030, 26.3037186
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5416870, 29.5411301
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7461929, 20.7451248
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6371765, 27.6354065
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5900249, 17.5898438
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4342957, 34.4305344
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0242615, 31.0229340
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5782471, 39.5805588
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6944199, 21.6978035
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2470932, 25.2390518
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0261536, 27.0275116
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7151871, 14.7146034
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4412994, 15.4432449
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4410248, 18.4405670
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3596420, 18.3650131
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8035126, 15.8058662
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4683723, 14.4714050
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4821205, 15.4914818
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7549591, 19.7589455
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6454048, 17.6477585
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3113518, 16.3173923
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7896500, 17.7897034
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8165932, 18.8198090
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6477890, 22.6413498
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8673401, 28.8578949
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9653244, 22.9573174
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9409103, 24.9331970
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4198151, 31.4177322
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5810089, 28.5690994
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8590240, 33.8484268
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9326515, 19.9181633
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3538551, 21.3442841
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3884087, 15.3867531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1660

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9114474, upper bound: 9.9407956
time: 24.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9164301, upper bound: 9.9361927
time: 23.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6885681, 31.6897964
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7775192, 18.7756119
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5190887, 23.5182877
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4093552, 22.4060364
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1341629, 24.1347046
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8678856, 19.8618774
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3066711, 26.3026428
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5425568, 29.5402679
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7464752, 20.7448425
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6362991, 27.6362839
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5918636, 17.5880051
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4329987, 34.4318390
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0243835, 31.0228043
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5792542, 39.5795517
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6936722, 21.6985550
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2472305, 25.2389107
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0273895, 27.0262756
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7157021, 14.7140846
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4423866, 15.4421577
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4422913, 18.4393005
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3587875, 18.3658638
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8033600, 15.8060188
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4691200, 14.4706573
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4829979, 15.4906082
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7567062, 19.7571945
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6454582, 17.6477089
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3112717, 16.3174706
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7898026, 17.7895584
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8176003, 18.8188057
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6464996, 22.6426392
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8632355, 28.8619995
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9610825, 22.9615593
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9379120, 24.9362030
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4195709, 31.4179764
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5784607, 28.5716400
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8576813, 33.8497620
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9303322, 19.9204826
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3527489, 21.3453827
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3877373, 15.3874283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1365

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1301

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9178641, upper bound: 9.9432335
time: 19.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9185869, upper bound: 9.9425109
time: 22.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6854248, 31.6841888
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7686920, 18.7675285
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5158463, 23.5145493
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.3856506, 22.3834686
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1274872, 24.1284256
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8238220, 19.8167839
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3104858, 26.3090286
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5356140, 29.5334091
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7387009, 20.7353973
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6408615, 27.6399460
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5918007, 17.5947456
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4140778, 34.4116669
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0174103, 31.0139008
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5646362, 39.5669403
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6782684, 21.6801224
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2462387, 25.2378998
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0261917, 27.0290680
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7156620, 14.7160320
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4302101, 15.4329185
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4428940, 18.4442978
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3722267, 18.3779335
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8047714, 15.8070259
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4584656, 14.4624023
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4853477, 15.4933777
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7551422, 19.7576447
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6446228, 17.6473885
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3099861, 16.3166885
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7669716, 17.7698288
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8246346, 18.8278122
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6307907, 22.6239853
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8845901, 28.8725662
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9544907, 22.9483643
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9521561, 24.9462204
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4180603, 31.4154358
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5632935, 28.5489349
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8768616, 33.8644943
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9069366, 19.8892097
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3221436, 21.3116570
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3576546, 15.3522949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1693

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 647

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.8964293, upper bound: 9.9290010
time: 14.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9046976, upper bound: 9.9207248
time: 23.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6855469, 31.6840439
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7674561, 18.7687721
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5155563, 23.5148315
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.3832016, 22.3859177
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1265717, 24.1293335
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8291016, 19.8114967
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3106537, 26.3088531
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5344543, 29.5345764
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7378540, 20.7362442
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6408005, 27.6400070
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5943260, 17.5922165
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4166718, 34.4090881
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0146637, 31.0166397
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5621338, 39.5694427
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6760941, 21.6822968
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2461929, 25.2379456
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0262299, 27.0290298
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7141476, 14.7175446
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4305077, 15.4326210
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4420395, 18.4451523
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3700523, 18.3801041
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8047943, 15.8070068
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4580917, 14.4627800
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4830971, 15.4956245
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7556000, 19.7571831
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6449356, 17.6470795
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3099670, 16.3167076
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7695198, 17.7672768
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8218880, 18.8305626
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6342316, 22.6205444
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8857651, 28.8713989
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9595032, 22.9433517
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9525299, 24.9458466
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4184875, 31.4150009
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5659256, 28.5463028
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8763733, 33.8649902
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9109344, 19.8852234
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3265228, 21.3072739
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3621216, 15.3478317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1789

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 775

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9171360, upper bound: 9.9409303
time: 26.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9152501, upper bound: 9.9428114
time: 27.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 56.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9386853, upper bound: 9.9229875
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9393568, upper bound: 9.9223153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9369691, upper bound: 9.9304602
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9394633, upper bound: 9.9279815
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9355296, upper bound: 9.9294148
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9405411, upper bound: 9.9244050
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9114474, upper bound: 9.9407956
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9164301, upper bound: 9.9361927
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9178641, upper bound: 9.9432335
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9185869, upper bound: 9.9425109
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.8964293, upper bound: 9.9290010
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9046976, upper bound: 9.9207248
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9171360, upper bound: 9.9409303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 56.60
Output dim: 25, lower bound: -9.9152501, upper bound: 9.9428114

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6551361, 31.6509094
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7789001, 18.7792892
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5199356, 23.5208206
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4164429, 22.4201660
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1346283, 24.1343536
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8331833, 19.8479805
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3033829, 26.3073120
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5447388, 29.5461960
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7217255, 20.7209816
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6348038, 27.6348419
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5969429, 17.5988541
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4349670, 34.4396362
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0141296, 31.0144958
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5830688, 39.5796661
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6924591, 21.6852493
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2306824, 25.2393265
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0215836, 27.0203629
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7156525, 14.7153893
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4139595, 15.4146729
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4346428, 18.4346352
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3586426, 18.3505821
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8129463, 15.8069687
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4535751, 14.4509926
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4711456, 15.4606819
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7475853, 19.7464409
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6472778, 17.6443901
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3029366, 16.2963867
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7593231, 17.7631149
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8035507, 18.8000259
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6457672, 22.6545639
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8472443, 28.8546677
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9757309, 22.9837303
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9327393, 24.9378357
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4134674, 31.4161148
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5117188, 28.5209427
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8253937, 33.8353577
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9010086, 19.9187851
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3151588, 21.3265762
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3691540, 15.3727989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1694

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 948

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9275185, upper bound: 9.9223789
time: 21.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9380783, upper bound: 9.9118154
time: 23.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6549988, 31.6510468
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7788849, 18.7793045
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5199356, 23.5208206
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4163437, 22.4202652
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1345978, 24.1343689
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8331757, 19.8479843
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3033218, 26.3073578
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5447235, 29.5462189
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7216339, 20.7210808
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6348190, 27.6348267
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5969429, 17.5988541
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4349670, 34.4396439
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0140991, 31.0145264
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5830688, 39.5796509
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6925125, 21.6851959
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2306137, 25.2393951
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0216064, 27.0203400
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7156296, 14.7154121
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4139290, 15.4146996
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4345665, 18.4347153
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3587112, 18.3505173
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8130150, 15.8069000
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4535980, 14.4509735
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4711838, 15.4606400
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7475853, 19.7464409
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6472855, 17.6443787
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3029671, 16.2963600
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7594070, 17.7630310
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8036118, 18.7999611
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6457443, 22.6545792
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8470612, 28.8548508
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9757614, 22.9836998
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9327087, 24.9378662
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4134521, 31.4161224
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5114594, 28.5212021
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8253174, 33.8354340
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9008255, 19.9189644
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3150902, 21.3266487
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3688488, 15.3731003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1309

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 985

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9386001, upper bound: 9.9203213
time: 26.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9373622, upper bound: 9.9215596
time: 22.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6167450, 31.6176453
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7768631, 18.7772331
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5262985, 23.5264282
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4100800, 22.4125900
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1538544, 24.1528168
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.7900772, 19.7940865
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3287201, 26.3288422
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5488281, 29.5496140
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6720123, 20.6750565
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6319809, 27.6324081
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5631981, 17.5593185
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4338684, 34.4366531
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9888611, 30.9947281
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5921478, 39.5916748
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6593933, 21.6589813
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2091217, 25.2177658
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0098038, 27.0084229
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7202263, 14.7205849
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.3687096, 15.3652382
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4370155, 18.4359245
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3540459, 18.3507957
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8213844, 15.8182259
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4608078, 14.4590263
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4882622, 15.4836693
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7155418, 19.7112885
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6488228, 17.6469688
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2924309, 16.2873840
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7521172, 17.7503662
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8048058, 18.8028183
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6316528, 22.6352081
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7694092, 28.7803116
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9751205, 22.9800377
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.8928452, 24.8995514
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4119415, 31.4142303
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.4697113, 28.4823990
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.7734833, 33.7886200
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9109344, 19.9244957
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3184166, 21.3251610
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3741589, 15.3766060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 807

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1685

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9386396, upper bound: 9.9235937
time: 29.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9350759, upper bound: 9.9271587
time: 21.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6492767, 31.6441498
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7826233, 18.7826729
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5205612, 23.5214767
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4147797, 22.4170990
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1392212, 24.1393433
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8059425, 19.8207893
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3291473, 26.3337250
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5540771, 29.5551376
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7101288, 20.7084198
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6348724, 27.6345367
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5697727, 17.5759659
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4364166, 34.4399414
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0279694, 31.0265121
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5840454, 39.5820007
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6757660, 21.6681213
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2463684, 25.2526779
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0083160, 27.0076218
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7168312, 14.7167397
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.3867722, 15.3895836
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4382362, 18.4398804
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3885880, 18.3809204
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8123283, 15.8076744
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4569931, 14.4554329
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4813385, 15.4738388
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7266006, 19.7276154
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6470947, 17.6450348
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3099709, 16.3053360
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7704086, 17.7770424
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8032455, 18.8009453
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6450806, 22.6513062
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8022461, 28.8030777
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9741974, 22.9783859
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9024506, 24.9038239
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4134216, 31.4152451
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5149155, 28.5184784
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8251953, 33.8282928
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9085655, 19.9207573
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3192673, 21.3268776
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3712673, 15.3734169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1710

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9311948, upper bound: 9.9239649
time: 16.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9401064, upper bound: 9.9150782
time: 21.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6855927, 31.6852341
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7743530, 18.7723999
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5180130, 23.5169983
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4087219, 22.4054871
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1337738, 24.1345825
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8635826, 19.8581276
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3028564, 26.2993927
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5378723, 29.5350723
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7401352, 20.7355270
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6366577, 27.6357498
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5892601, 17.5884895
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4321594, 34.4289856
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0230713, 31.0204239
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5788422, 39.5788345
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6940231, 21.6977844
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2408447, 25.2291603
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0250778, 27.0264740
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7147026, 14.7139816
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4413872, 15.4424019
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4376411, 18.4353790
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3540039, 18.3613777
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8034134, 15.8056107
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4639778, 14.4677200
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4821396, 15.4914513
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7551003, 19.7587318
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6448250, 17.6471786
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3083267, 16.3163548
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7840958, 17.7862587
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8165169, 18.8191414
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6449738, 22.6391907
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8583908, 28.8515396
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9450989, 22.9439316
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9327011, 24.9280090
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4189301, 31.4170685
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5749588, 28.5648651
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8589478, 33.8483810
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9295006, 19.9160309
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3486900, 21.3405952
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3874855, 15.3859177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1337

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1623

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9109198, upper bound: 9.9285833
time: 39.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.8994708, upper bound: 9.9400292
time: 24.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6862793, 31.6872253
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7774811, 18.7755547
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5191116, 23.5183563
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4098053, 22.4064789
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1335907, 24.1341934
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8684998, 19.8624306
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3070068, 26.3029327
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5424957, 29.5401764
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7465286, 20.7447891
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6366043, 27.6366272
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5930023, 17.5890274
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4325409, 34.4313202
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0255890, 31.0241928
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5795135, 39.5798645
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6935272, 21.6984673
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2477570, 25.2393608
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0272141, 27.0260773
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7145824, 14.7129917
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4424591, 15.4422321
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4424057, 18.4394073
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3581696, 18.3654823
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8030968, 15.8055496
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4706573, 14.4723701
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4844513, 15.4922447
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7573891, 19.7577744
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6454315, 17.6476898
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3103561, 16.3166618
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7874985, 17.7874413
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8158836, 18.8172493
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6479034, 22.6438370
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8635406, 28.8622742
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9612274, 22.9617233
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9387741, 24.9371033
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4194489, 31.4178314
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5771255, 28.5700836
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8588562, 33.8507767
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9313965, 19.9213295
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3521729, 21.3445206
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3891830, 15.3883133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 647

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1381

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9177838, upper bound: 9.9423011
time: 27.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9169336, upper bound: 9.9431532
time: 20.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6860046, 31.6875000
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7774658, 18.7755775
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5191650, 23.5183105
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4097900, 22.4064941
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1336517, 24.1341400
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8684387, 19.8624878
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3069611, 26.3029709
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5424500, 29.5402222
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7464218, 20.7448959
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6366348, 27.6366043
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5928879, 17.5891457
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4324799, 34.4313736
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0257721, 31.0240173
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5795746, 39.5798035
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6935883, 21.6984138
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2476807, 25.2394371
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0271835, 27.0261078
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7146091, 14.7129631
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4424591, 15.4422302
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4423981, 18.4394150
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3584061, 18.3652458
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8028908, 15.8057518
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4708405, 14.4721909
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4846420, 15.4920616
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7572899, 19.7578735
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6454391, 17.6476822
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3104630, 16.3165569
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7876816, 17.7872543
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8160439, 18.8170891
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6476898, 22.6440430
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8635101, 28.8623047
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9612503, 22.9617043
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9388199, 24.9370575
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4194336, 31.4178467
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5769043, 28.5703049
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8587036, 33.8509293
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9311829, 19.9215469
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3518906, 21.3448067
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3886185, 15.3888779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1714

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9179180, upper bound: 9.9343145
time: 22.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9103916, upper bound: 9.9418414
time: 18.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6853943, 31.6845779
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7668304, 18.7677574
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5153961, 23.5149612
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.3805695, 22.3820114
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1263580, 24.1290665
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8252335, 19.8095055
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3100739, 26.3079758
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5343933, 29.5345154
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7378082, 20.7366333
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6400604, 27.6392365
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5945797, 17.5919228
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4139404, 34.4072113
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0137024, 31.0154114
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5617676, 39.5682831
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6760406, 21.6822357
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2459030, 25.2378845
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0267487, 27.0286484
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7130451, 14.7154083
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4300728, 15.4319687
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4404640, 18.4426918
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3694534, 18.3789253
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8044968, 15.8068161
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4585533, 14.4622459
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4824810, 15.4944801
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7555466, 19.7571564
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6447754, 17.6467743
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3101730, 16.3166733
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7692757, 17.7667503
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8200760, 18.8276749
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6334839, 22.6213684
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8828888, 28.8702927
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9578857, 22.9435501
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9525070, 24.9459991
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4184418, 31.4152679
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5617599, 28.5437393
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8763733, 33.8651657
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9043427, 19.8810005
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3227768, 21.3051987
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3549957, 15.3436432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1321

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9078057, upper bound: 9.9404987
time: 13.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9167013, upper bound: 9.9315857
time: 25.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6860962, 31.6838760
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7664490, 18.7681465
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5156860, 23.5146637
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.3792953, 22.3832779
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1263123, 24.1291199
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8271179, 19.8076248
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3097839, 26.3082657
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5343933, 29.5345230
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7382507, 20.7361984
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6400223, 27.6392670
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5940380, 17.5924683
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4147949, 34.4063568
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0134277, 31.0156784
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5609894, 39.5690613
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6760330, 21.6822433
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2461395, 25.2376480
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0258484, 27.0295563
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7120152, 14.7164383
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4298553, 15.4321842
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4395790, 18.4435768
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3688736, 18.3795052
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8046036, 15.8067055
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4575577, 14.4632416
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4819546, 15.4950066
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7555695, 19.7571297
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6446304, 17.6469193
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3099365, 16.3169117
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7689934, 17.7670288
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8190002, 18.8287468
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6350555, 22.6197968
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8846588, 28.8685226
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9597015, 22.9417419
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9526825, 24.9458237
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4187622, 31.4149551
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5633621, 28.5421295
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8765564, 33.8649826
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9067078, 19.8786354
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3244476, 21.3035278
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3579330, 15.3407059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1315

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9145211, upper bound: 9.9427552
time: 27.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9151938, upper bound: 9.9420829
time: 24.44 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 53.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9275185, upper bound: 9.9223789
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9380783, upper bound: 9.9118154
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9386001, upper bound: 9.9203213
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9373622, upper bound: 9.9215596
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9386396, upper bound: 9.9235937
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9350759, upper bound: 9.9271587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9311948, upper bound: 9.9239649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9401064, upper bound: 9.9150782
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9109198, upper bound: 9.9285833
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.8994708, upper bound: 9.9400292
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9177838, upper bound: 9.9423011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9169336, upper bound: 9.9431532
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9179180, upper bound: 9.9343145
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9103916, upper bound: 9.9418414
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9078057, upper bound: 9.9404987
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9167013, upper bound: 9.9315857
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9145211, upper bound: 9.9427552
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 53.72
Output dim: 25, lower bound: -9.9151938, upper bound: 9.9420829

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6546173, 31.6501465
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7776718, 18.7794800
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5198517, 23.5207977
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4136124, 22.4194489
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1345367, 24.1342316
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8302956, 19.8461075
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3018494, 26.3078842
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5428467, 29.5457840
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7210388, 20.7204933
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6338425, 27.6332245
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5963707, 17.5980034
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4350433, 34.4394302
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0138092, 31.0144348
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5822144, 39.5774536
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6918716, 21.6832542
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2302322, 25.2401123
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0215302, 27.0202713
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7155914, 14.7153034
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4139519, 15.4146290
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4341240, 18.4336128
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3573494, 18.3474464
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8117256, 15.8049011
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4529800, 14.4499207
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4687729, 15.4550400
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7471962, 19.7462387
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6465874, 17.6429634
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3013725, 16.2928810
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7589111, 17.7621994
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8035316, 18.8000031
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6420670, 22.6526260
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8441620, 28.8530426
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9751282, 22.9832993
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9326477, 24.9377747
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4127045, 31.4156952
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5081635, 28.5193100
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8195648, 33.8324738
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.8933487, 19.9150963
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3108177, 21.3245087
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3690948, 15.3727531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1395

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1547

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9369573, upper bound: 9.9112676
time: 19.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9373815, upper bound: 9.9100772
time: 24.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6551666, 31.6511917
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7718811, 18.7730560
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5149155, 23.5164795
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4084854, 22.4133835
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1282196, 24.1288300
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8314133, 19.8462181
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.2987137, 26.3031616
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5367889, 29.5393372
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7178192, 20.7176552
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6340790, 27.6336517
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5919781, 17.5926132
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4308472, 34.4349060
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0145416, 31.0149765
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5802155, 39.5771027
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6912689, 21.6839638
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2303238, 25.2390900
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0190201, 27.0173569
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7153664, 14.7150650
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4142265, 15.4155045
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4345322, 18.4346504
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3588905, 18.3508797
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8116684, 15.8048439
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4486847, 14.4449539
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4682064, 15.4575920
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7506256, 19.7499161
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6457596, 17.6425323
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3010597, 16.2943230
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7556229, 17.7578545
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8049889, 18.8020020
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6505585, 22.6596985
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8463821, 28.8536682
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9709778, 22.9771614
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9321899, 24.9362793
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4127655, 31.4152832
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.4991379, 28.5069962
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8250427, 33.8350601
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.8985367, 19.9166222
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3099365, 21.3209229
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3672466, 15.3714333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 808

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1310

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9383868, upper bound: 9.9203101
time: 21.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9385890, upper bound: 9.9201071
time: 17.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6551361, 31.6512146
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7726288, 18.7723083
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5155869, 23.5158005
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4094620, 22.4124069
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1290741, 24.1279831
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8314133, 19.8462181
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.2991257, 26.3027420
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5378418, 29.5382919
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7182007, 20.7172737
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6336441, 27.6340866
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5907040, 17.5938873
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4302216, 34.4355392
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0145416, 31.0149841
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5805054, 39.5768127
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6912766, 21.6839523
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2303009, 25.2391090
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0186234, 27.0177460
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7152824, 14.7151508
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4147339, 15.4149971
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4345016, 18.4346848
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3590736, 18.3507042
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8109589, 15.8055496
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4475784, 14.4460602
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4681339, 15.4576645
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7510605, 19.7494812
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6454391, 17.6428528
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3009300, 16.2944527
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7542343, 17.7592430
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8056526, 18.8013382
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6508636, 22.6593933
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8458786, 28.8541718
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9692230, 22.9789200
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9311218, 24.9373474
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4126129, 31.4154358
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.4972458, 28.5088806
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8249512, 33.8351593
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.8984833, 19.9166794
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3093643, 21.3214951
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3671856, 15.3714943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 663

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1318

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9372110, upper bound: 9.9196134
time: 21.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9354156, upper bound: 9.9214085
time: 13.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6187897, 31.6197433
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7858353, 18.7899132
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5168533, 23.5175705
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4172592, 22.4264374
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1570969, 24.1567307
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.7761650, 19.7815819
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3354950, 26.3377991
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5552368, 29.5570984
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6458435, 20.6457863
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.5948792, 27.5906372
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5410366, 17.5335770
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4415588, 34.4411163
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9957123, 31.0002823
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5417938, 39.5350189
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6547012, 21.6541557
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2020187, 25.2095528
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0074615, 27.0067215
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7246265, 14.7241592
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.3583107, 15.3539124
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4185410, 18.4146423
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3534584, 18.3501205
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8124199, 15.8082657
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4615593, 14.4587364
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4862080, 15.4769440
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7234612, 19.7222633
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6463089, 17.6433220
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2842331, 16.2774849
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7355690, 17.7309151
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8093491, 18.8074570
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6218872, 22.6270828
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7165451, 28.7327881
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9671707, 22.9729347
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.8898392, 24.8966522
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4019012, 31.4055634
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.4104538, 28.4301300
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.7047272, 33.7277527
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.8459129, 19.8666573
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.2726593, 21.2842102
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3683949, 15.3729362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9277357, upper bound: 9.9197521
time: 22.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9348893, upper bound: 9.9125750
time: 22.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6391754, 31.6352158
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7720108, 18.7733650
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5104218, 23.5119629
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4137878, 22.4161224
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1312103, 24.1318359
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8142242, 19.8288879
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3128052, 26.3192368
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5347137, 29.5380096
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6923981, 20.6927223
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6384354, 27.6377869
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5734901, 17.5803528
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4347382, 34.4381027
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0350494, 31.0343170
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5931854, 39.5923767
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6866913, 21.6781273
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2182693, 25.2278442
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0055847, 27.0048218
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7160168, 14.7159424
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.3885460, 15.3918571
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4320335, 18.4349442
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3684731, 18.3580055
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8127899, 15.8081589
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4448586, 14.4421997
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4721355, 15.4644203
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7275238, 19.7288933
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6406288, 17.6380920
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2930984, 16.2861271
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7703667, 17.7749977
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8041039, 18.8018646
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6502838, 22.6564560
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7913437, 28.7911911
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9427872, 22.9433594
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.8879700, 24.8876266
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4154053, 31.4172134
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5114365, 28.5147247
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8333893, 33.8368454
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9158745, 19.9280357
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3228455, 21.3304062
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3812199, 15.3840027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1604

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9369160, upper bound: 9.9142052
time: 22.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9392329, upper bound: 9.9118875
time: 25.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6811676, 31.6808624
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7628098, 18.7624054
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5075989, 23.5051193
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4020844, 22.3998337
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1297073, 24.1334305
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8435364, 19.8359528
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3002930, 26.2965622
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5415344, 29.5383377
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.6946640, 20.6840591
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6434326, 27.6390457
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5840931, 17.5837860
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4178619, 34.4089813
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -30.9880219, 30.9807892
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5924530, 39.5923309
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6945877, 21.6985397
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2400055, 25.2223892
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -26.9668884, 26.9755630
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.6915779, 14.6937485
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4471359, 15.4501991
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4321899, 18.4312057
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3439560, 18.3536415
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.7778625, 15.7830048
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4244194, 14.4326820
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4725418, 15.4827499
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.6984444, 19.7085152
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6418037, 17.6444817
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2918282, 16.3015842
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7887802, 17.7913818
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.7856369, 18.7923050
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6158829, 22.6071320
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.7970200, 28.7826080
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.8969269, 22.8883896
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.8896942, 24.8792953
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4154053, 31.4131775
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5569305, 28.5433884
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8142242, 33.7988510
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.8789444, 19.8593369
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3328476, 21.3232536
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3585167, 15.3514290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1574

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 854

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.8926419, upper bound: 9.9395911
time: 21.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.8990427, upper bound: 9.9302023
time: 22.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6872711, 31.6884155
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7792892, 18.7773819
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5166855, 23.5155792
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4120712, 22.4088058
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1338196, 24.1344070
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8641930, 19.8586235
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3097076, 26.3060913
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5429077, 29.5405426
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7437439, 20.7417564
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6322250, 27.6315155
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5933075, 17.5893364
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4336700, 34.4324875
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0254059, 31.0236893
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5750427, 39.5747375
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6902161, 21.6947136
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2484360, 25.2399597
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0206375, 27.0200577
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7136879, 14.7119102
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4418106, 15.4416237
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4424210, 18.4394226
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3607178, 18.3677406
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8030357, 15.8054962
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4703369, 14.4721107
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4894829, 15.4965515
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7529144, 19.7538109
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6455879, 17.6478996
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3083458, 16.3144569
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7880745, 17.7880173
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8152504, 18.8165588
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6456146, 22.6418228
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8618011, 28.8607559
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9622192, 22.9629517
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9398117, 24.9380798
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4180908, 31.4166641
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5681763, 28.5619278
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8550568, 33.8472824
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9237061, 19.9145012
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3461456, 21.3392601
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3889694, 15.3882370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 917

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1365

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9176718, upper bound: 9.9411945
time: 24.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9163894, upper bound: 9.9420744
time: 23.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6874542, 31.6882172
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7793121, 18.7773514
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5163345, 23.5159302
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4121323, 22.4087448
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1337891, 24.1344376
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8646812, 19.8581276
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3101501, 26.3056412
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5428772, 29.5405807
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7434845, 20.7420082
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6315002, 27.6322479
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5933151, 17.5893288
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4337006, 34.4324570
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0251007, 31.0240021
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5743713, 39.5754089
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6897736, 21.6951485
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2483597, 25.2400360
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0212021, 27.0194931
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7134972, 14.7121010
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4418526, 15.4415836
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4424210, 18.4394264
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3604279, 18.3680344
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8030357, 15.8054962
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4703941, 14.4720535
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4887581, 15.4972801
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7534180, 19.7533112
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6456413, 17.6478500
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3081512, 16.3146515
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7880745, 17.7880173
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8151894, 18.8166199
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6458893, 22.6415482
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8620300, 28.8605347
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9624481, 22.9627151
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9397430, 24.9381409
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4182739, 31.4164810
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5689697, 28.5611343
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8553619, 33.8469696
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9245682, 19.9136391
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3469086, 21.3384972
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3891068, 15.3880978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1395

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1781

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9164508, upper bound: 9.9424944
time: 15.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9162790, upper bound: 9.9426674
time: 22.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6858521, 31.6871185
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7774353, 18.7746773
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5195007, 23.5181732
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4085922, 22.4041748
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1336365, 24.1341248
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8649902, 19.8584633
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3066254, 26.3015442
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5429382, 29.5394287
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7454376, 20.7427750
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6361465, 27.6361465
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5920372, 17.5883942
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4315491, 34.4304352
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0244751, 31.0218735
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5790100, 39.5792236
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6905823, 21.6960678
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2438812, 25.2333565
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0267258, 27.0259781
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7145596, 14.7129135
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4416656, 15.4415627
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4422913, 18.4392853
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3538246, 18.3622932
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.7991409, 15.8027534
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4689217, 14.4714661
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4757042, 15.4850273
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7565155, 19.7572021
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6426315, 17.6456642
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3035507, 16.3115063
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7830200, 17.7839279
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8153458, 18.8164291
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6443863, 22.6394424
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8577194, 28.8551331
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9610367, 22.9616013
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9376144, 24.9356461
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4190369, 31.4170074
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5712280, 28.5631714
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8498535, 33.8392181
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9210968, 19.9092102
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3473053, 21.3389435
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3877106, 15.3877201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1781

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1582

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9100543, upper bound: 9.9405753
time: 25.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9091252, upper bound: 9.9415036
time: 32.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6764679, 31.6744919
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7575073, 18.7571373
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5058823, 23.5048218
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.3795853, 22.3810120
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1188507, 24.1210556
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8333321, 19.8177795
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.2955856, 26.2916336
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5172577, 29.5151443
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7221222, 20.7189064
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6432877, 27.6427841
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5989723, 17.5956459
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4121094, 34.4055252
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0215149, 31.0224991
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5721588, 39.5774536
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6860504, 21.6931610
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2210693, 25.2097969
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0239487, 27.0259171
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7122440, 14.7145901
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4323502, 15.4337463
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4355278, 18.4364929
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3465385, 18.3588028
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8049698, 15.8072624
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4453201, 14.4501114
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4730740, 15.4852905
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7568283, 19.7580872
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6378403, 17.6403198
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.2909737, 16.2998104
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7672234, 17.7667084
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8209877, 18.8285255
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6386261, 22.6265640
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8710098, 28.8594055
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9228897, 22.9121552
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9363098, 24.9315262
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4204254, 31.4172668
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5579987, 28.5402679
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8849487, 33.8733673
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9116249, 19.8883018
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3263245, 21.3087883
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3655777, 15.3535919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1354

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1317

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9076559, upper bound: 9.9403572
time: 22.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9076648, upper bound: 9.9403483
time: 21.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6853485, 31.6829834
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7664337, 18.7681122
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5156784, 23.5146561
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.3792038, 22.3830948
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1258850, 24.1286850
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8271484, 19.8076553
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3095093, 26.3079529
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5345612, 29.5346832
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7388916, 20.7367439
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6400986, 27.6393509
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5938702, 17.5923080
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4146881, 34.4062500
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0131378, 31.0153503
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5610046, 39.5690765
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6764297, 21.6826897
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2467422, 25.2381897
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0251389, 27.0288696
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7115135, 14.7159157
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4293594, 15.4316597
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4389038, 18.4428177
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3686600, 18.3793526
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8050995, 15.8072739
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4574966, 14.4631996
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4818707, 15.4949646
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7554932, 19.7570534
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6446457, 17.6469460
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3096771, 16.3166809
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7685814, 17.7667007
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8184738, 18.8282852
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6349106, 22.6196365
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8839417, 28.8676224
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9594498, 22.9415207
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9528275, 24.9459381
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4186401, 31.4148254
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5621719, 28.5406799
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8760223, 33.8643799
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9067230, 19.8784752
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3239136, 21.3029251
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3559875, 15.3384552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1461

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9124477, upper bound: 9.9308091
time: 25.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9074016, upper bound: 9.9408033
time: 25.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6852112, 31.6831284
1: -12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314
2: -11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7664185, 18.7681351
3: -17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5156784, 23.5146561
4: -19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.3791122, 22.3831863
5: -15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1258545, 24.1287003
6: -31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8271484, 19.8076630
7: -21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3094635, 26.3079910
8: -23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5345459, 29.5346985
9: -13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7388000, 20.7368355
10: -13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6401062, 27.6393433
11: -10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.5938702, 17.5923042
12: -23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4146729, 34.4062653
13: -25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0131073, 31.0153885
14: -26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5610046, 39.5690765
15: -10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6764832, 21.6826363
16: -20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2466736, 25.2382584
17: -23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835
18: -11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0251617, 27.0288544
19: -7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7114906, 14.7159386
20: -6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4293327, 15.4316883
21: -7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4388199, 18.4429016
22: -5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3687286, 18.3792915
23: -2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8051758, 15.8072052
24: -5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4575195, 14.4631805
25: -0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4819088, 15.4949265
26: -12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547
27: -9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7554932, 19.7570572
28: -4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6446533, 17.6469383
29: -3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3097038, 16.3166542
30: -10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7686653, 17.7666168
31: -6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8185349, 18.8282204
32: -26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6348953, 22.6196518
33: -43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.8837585, 28.8678131
34: -36.2045822, -6.0280871, -36.2045822, -6.0280871, -22.9594803, 22.9414940
35: -26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9527969, 24.9459610
36: -27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4186249, 31.4148407
37: -44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.5619125, 28.5409393
38: -31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414
39: -48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8759613, 33.8644485
40: -44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9065399, 19.8786545
41: -30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3238449, 21.3029938
42: -19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.3556862, 15.3387585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1332

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1299

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9130904, upper bound: 9.9420507
time: 28.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9151616, upper bound: 9.9402965
time: 21.13 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 51.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9369573, upper bound: 9.9112676
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9373815, upper bound: 9.9100772
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9383868, upper bound: 9.9203101
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9385890, upper bound: 9.9201071
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9372110, upper bound: 9.9196134
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9354156, upper bound: 9.9214085
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9277357, upper bound: 9.9197521
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9348893, upper bound: 9.9125750
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9369160, upper bound: 9.9142052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9392329, upper bound: 9.9118875
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.8926419, upper bound: 9.9395911
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.8990427, upper bound: 9.9302023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9176718, upper bound: 9.9411945
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9163894, upper bound: 9.9420744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9164508, upper bound: 9.9424944
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9162790, upper bound: 9.9426674
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9100543, upper bound: 9.9405753
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9091252, upper bound: 9.9415036
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9076559, upper bound: 9.9403572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9076648, upper bound: 9.9403483
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9124477, upper bound: 9.9308091
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9074016, upper bound: 9.9408033
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9130904, upper bound: 9.9420507
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 51.79
Output dim: 25, lower bound: -9.9151616, upper bound: 9.9402965

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 43.25 + 1772.37 = 1815.61 seconds
