## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 7)
Time budget: 7200 seconds
Split limit: 100
Threshold: 15.0175617858


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=200, inp2_unstable=200, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.8484058, 27.0876694, -5.8484058, 27.0876694, -32.9360733, 32.9360733)
1: (0.0080841, 22.9084625, 0.0080841, 22.9084625, -20.6950302, 20.6950302)
2: (-0.9063466, 25.1065483, -0.9063466, 25.1065483, -23.6805115, 23.6805038)
3: (-12.6358604, 15.9553623, -12.6358604, 15.9553623, -23.3256836, 23.3256836)
4: (-6.2847629, 20.8345890, -6.2847629, 20.8345890, -23.0718307, 23.0718307)
5: (-11.9136658, 20.9306450, -11.9136658, 20.9306450, -30.6666794, 30.6666641)
6: (-66.2194138, -30.6795292, -66.2194138, -30.6795292, -24.8722992, 24.8722992)
7: (-17.8853855, 14.9396420, -17.8853855, 14.9396420, -27.4274673, 27.4274673)
8: (-19.3295956, 10.2857838, -19.3295956, 10.2857838, -24.6739426, 24.6739464)
9: (-7.1940708, 22.8957558, -7.1940708, 22.8957558, -29.1299667, 29.1299515)
10: (-31.6227436, 12.2977619, -31.6227436, 12.2977619, -39.1584473, 39.1584549)
11: (-24.7147980, 0.1991603, -24.7147980, 0.1991603, -23.4555664, 23.4555664)
12: (-49.0830994, -8.5243492, -49.0830994, -8.5243492, -34.1352692, 34.1352692)
13: (-28.5756912, 19.1810932, -28.5756912, 19.1810932, -47.7567825, 47.7567825)
14: (-41.8362350, 8.8776398, -41.8362350, 8.8776398, -44.1171722, 44.1171799)
15: (-1.2438402, 28.1130295, -1.2438402, 28.1130295, -24.7512436, 24.7512436)
16: (-26.6712437, 10.5420656, -26.6712437, 10.5420656, -34.7192841, 34.7192764)
17: (-39.8865776, 4.8747683, -39.8865776, 4.8747683, -36.0841446, 36.0841446)
18: (-9.7624674, 26.5586090, -9.7624674, 26.5586090, -36.3210754, 36.3210754)
19: (-17.4268913, 9.3083858, -17.4268913, 9.3083858, -26.7352772, 26.7352772)
20: (-25.4231682, 3.7468839, -25.4231682, 3.7468839, -28.8696213, 28.8696289)
21: (-22.0199394, 8.0764503, -22.0199394, 8.0764503, -30.0963898, 30.0963898)
22: (-10.4554958, 21.3130703, -10.4554958, 21.3130703, -29.7908020, 29.7908096)
23: (-10.3478737, 18.1226082, -10.3478737, 18.1226082, -28.2309341, 28.2309341)
24: (-6.4852304, 22.3751621, -6.4852304, 22.3751621, -28.2685852, 28.2685776)
25: (-9.5180445, 22.8397675, -9.5180445, 22.8397675, -32.2557983, 32.2557983)
26: (-23.2106190, 24.4033546, -23.2106190, 24.4033546, -46.9969940, 46.9969940)
27: (-15.9029713, 17.4331760, -15.9029713, 17.4331760, -33.3361473, 33.3361473)
28: (-13.9861851, 19.3545685, -13.9861851, 19.3545685, -31.5605164, 31.5605087)
29: (-10.9152431, 15.4316082, -10.9152431, 15.4316082, -23.1442947, 23.1442947)
30: (-30.3633671, -0.1006818, -30.3633671, -0.1006818, -27.8829346, 27.8829346)
31: (-18.1789322, 10.7383003, -18.1789322, 10.7383003, -28.9172325, 28.9172325)
32: (-51.2251053, -16.1040516, -51.2251053, -16.1040516, -27.8589020, 27.8589020)
33: (-69.0680237, -12.0135298, -69.0680237, -12.0135298, -49.9540253, 49.9540253)
34: (-63.1368065, -21.4582806, -63.1368065, -21.4582806, -29.6439819, 29.6439819)
35: (-42.8285522, -0.5206113, -42.8285522, -0.5206113, -34.4677429, 34.4677429)
36: (-42.2597694, 2.7720976, -42.2597694, 2.7720976, -36.3789673, 36.3789673)
37: (-75.2216797, -18.9945145, -75.2216797, -18.9945145, -41.9245911, 41.9245758)
38: (-52.3679466, 2.0344648, -52.3679466, 2.0344648, -47.3362122, 47.3362122)
39: (-72.2596970, -13.6542854, -72.2596970, -13.6542854, -54.5075226, 54.5074921)
40: (-76.4585266, -36.7105141, -76.4585266, -36.7105141, -28.9938965, 28.9938965)
41: (-52.0074844, -11.1838779, -52.0074844, -11.1838779, -29.2460327, 29.2460327)
42: (-47.7828827, -16.1841316, -47.7828827, -16.1841316, -24.5305176, 24.5305176)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.01 + 35.29 = 38.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 15, lower bound: -15.0476571, upper bound: 15.0476571

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1424
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1730

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0117923, upper bound: 15.0462015
time: 25.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0462015, upper bound: 15.0117923
time: 25.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 50.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 50.76
Output dim: 15, lower bound: -15.0117923, upper bound: 15.0462015
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 50.76
Output dim: 15, lower bound: -15.0462015, upper bound: 15.0117923

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.8484058, 27.0876694, -5.8484058, 27.0876694, -32.9360733, 32.9360733
1: 0.0080841, 22.9084625, 0.0080841, 22.9084625, -20.6941452, 20.6942139
2: -0.9063466, 25.1065483, -0.9063466, 25.1065483, -23.6772766, 23.6784515
3: -12.6358604, 15.9553623, -12.6358604, 15.9553623, -23.3183517, 23.3214569
4: -6.2847629, 20.8345890, -6.2847629, 20.8345890, -23.0664215, 23.0685501
5: -11.9136658, 20.9306450, -11.9136658, 20.9306450, -30.6633835, 30.6644592
6: -66.2194138, -30.6795292, -66.2194138, -30.6795292, -24.8684235, 24.8678741
7: -17.8853855, 14.9396420, -17.8853855, 14.9396420, -27.4236298, 27.4247284
8: -19.3295956, 10.2857838, -19.3295956, 10.2857838, -24.6645660, 24.6681480
9: -7.1940708, 22.8957558, -7.1940708, 22.8957558, -29.1302795, 29.1296997
10: -31.6227436, 12.2977619, -31.6227436, 12.2977619, -39.1578827, 39.1578217
11: -24.7147980, 0.1991603, -24.7147980, 0.1991603, -23.4537582, 23.4536629
12: -49.0830994, -8.5243492, -49.0830994, -8.5243492, -34.1326523, 34.1309662
13: -28.5756912, 19.1810932, -28.5756912, 19.1810932, -47.7567825, 47.7567825
14: -41.8362350, 8.8776398, -41.8362350, 8.8776398, -44.1101837, 44.1128693
15: -1.2438402, 28.1130295, -1.2438402, 28.1130295, -24.7457809, 24.7469902
16: -26.6712437, 10.5420656, -26.6712437, 10.5420656, -34.7191696, 34.7188034
17: -39.8865776, 4.8747683, -39.8865776, 4.8747683, -36.0781326, 36.0803070
18: -9.7624674, 26.5586090, -9.7624674, 26.5586090, -36.3210754, 36.3210754
19: -17.4268913, 9.3083858, -17.4268913, 9.3083858, -26.7352772, 26.7352772
20: -25.4231682, 3.7468839, -25.4231682, 3.7468839, -28.8678207, 28.8674469
21: -22.0199394, 8.0764503, -22.0199394, 8.0764503, -30.0963898, 30.0963898
22: -10.4554958, 21.3130703, -10.4554958, 21.3130703, -29.7907562, 29.7907791
23: -10.3478737, 18.1226082, -10.3478737, 18.1226082, -28.2291260, 28.2282486
24: -6.4852304, 22.3751621, -6.4852304, 22.3751621, -28.2685242, 28.2686539
25: -9.5180445, 22.8397675, -9.5180445, 22.8397675, -32.2557220, 32.2558746
26: -23.2106190, 24.4033546, -23.2106190, 24.4033546, -46.9969788, 46.9969788
27: -15.9029713, 17.4331760, -15.9029713, 17.4331760, -33.3361473, 33.3361473
28: -13.9861851, 19.3545685, -13.9861851, 19.3545685, -31.5585632, 31.5571899
29: -10.9152431, 15.4316082, -10.9152431, 15.4316082, -23.1427078, 23.1418610
30: -30.3633671, -0.1006818, -30.3633671, -0.1006818, -27.8822327, 27.8821030
31: -18.1789322, 10.7383003, -18.1789322, 10.7383003, -28.9172325, 28.9172325
32: -51.2251053, -16.1040516, -51.2251053, -16.1040516, -27.8581314, 27.8580780
33: -69.0680237, -12.0135298, -69.0680237, -12.0135298, -49.9538574, 49.9538116
34: -63.1368065, -21.4582806, -63.1368065, -21.4582806, -29.6438522, 29.6438065
35: -42.8285522, -0.5206113, -42.8285522, -0.5206113, -34.4662628, 34.4655075
36: -42.2597694, 2.7720976, -42.2597694, 2.7720976, -36.3757324, 36.3751068
37: -75.2216797, -18.9945145, -75.2216797, -18.9945145, -41.9207001, 41.9191132
38: -52.3679466, 2.0344648, -52.3679466, 2.0344648, -47.3327026, 47.3311310
39: -72.2596970, -13.6542854, -72.2596970, -13.6542854, -54.5070953, 54.5074310
40: -76.4585266, -36.7105141, -76.4585266, -36.7105141, -28.9925232, 28.9931183
41: -52.0074844, -11.1838779, -52.0074844, -11.1838779, -29.2421799, 29.2403259
42: -47.7828827, -16.1841316, -47.7828827, -16.1841316, -24.5301208, 24.5300598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=200, inp2_unstable=200, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1424
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9612361, upper bound: 15.0447218
time: 24.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0103613, upper bound: 14.9969975
time: 23.04 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.8484058, 27.0876694, -5.8484058, 27.0876694, -32.9360733, 32.9360733
1: 0.0080841, 22.9084625, 0.0080841, 22.9084625, -20.6942139, 20.6941452
2: -0.9063466, 25.1065483, -0.9063466, 25.1065483, -23.6784515, 23.6772766
3: -12.6358604, 15.9553623, -12.6358604, 15.9553623, -23.3214569, 23.3183517
4: -6.2847629, 20.8345890, -6.2847629, 20.8345890, -23.0685501, 23.0664177
5: -11.9136658, 20.9306450, -11.9136658, 20.9306450, -30.6644516, 30.6633835
6: -66.2194138, -30.6795292, -66.2194138, -30.6795292, -24.8678741, 24.8684273
7: -17.8853855, 14.9396420, -17.8853855, 14.9396420, -27.4247284, 27.4236298
8: -19.3295956, 10.2857838, -19.3295956, 10.2857838, -24.6681519, 24.6645699
9: -7.1940708, 22.8957558, -7.1940708, 22.8957558, -29.1296997, 29.1302795
10: -31.6227436, 12.2977619, -31.6227436, 12.2977619, -39.1578217, 39.1578827
11: -24.7147980, 0.1991603, -24.7147980, 0.1991603, -23.4536667, 23.4537544
12: -49.0830994, -8.5243492, -49.0830994, -8.5243492, -34.1309738, 34.1326523
13: -28.5756912, 19.1810932, -28.5756912, 19.1810932, -47.7567825, 47.7567825
14: -41.8362350, 8.8776398, -41.8362350, 8.8776398, -44.1128693, 44.1101761
15: -1.2438402, 28.1130295, -1.2438402, 28.1130295, -24.7469864, 24.7457771
16: -26.6712437, 10.5420656, -26.6712437, 10.5420656, -34.7188034, 34.7191849
17: -39.8865776, 4.8747683, -39.8865776, 4.8747683, -36.0803146, 36.0781250
18: -9.7624674, 26.5586090, -9.7624674, 26.5586090, -36.3210754, 36.3210754
19: -17.4268913, 9.3083858, -17.4268913, 9.3083858, -26.7352772, 26.7352772
20: -25.4231682, 3.7468839, -25.4231682, 3.7468839, -28.8674545, 28.8678131
21: -22.0199394, 8.0764503, -22.0199394, 8.0764503, -30.0963898, 30.0963898
22: -10.4554958, 21.3130703, -10.4554958, 21.3130703, -29.7907715, 29.7907562
23: -10.3478737, 18.1226082, -10.3478737, 18.1226082, -28.2282562, 28.2291336
24: -6.4852304, 22.3751621, -6.4852304, 22.3751621, -28.2686462, 28.2685318
25: -9.5180445, 22.8397675, -9.5180445, 22.8397675, -32.2558746, 32.2557220
26: -23.2106190, 24.4033546, -23.2106190, 24.4033546, -46.9969788, 46.9969788
27: -15.9029713, 17.4331760, -15.9029713, 17.4331760, -33.3361473, 33.3361473
28: -13.9861851, 19.3545685, -13.9861851, 19.3545685, -31.5571899, 31.5585556
29: -10.9152431, 15.4316082, -10.9152431, 15.4316082, -23.1418610, 23.1427078
30: -30.3633671, -0.1006818, -30.3633671, -0.1006818, -27.8821030, 27.8822327
31: -18.1789322, 10.7383003, -18.1789322, 10.7383003, -28.9172325, 28.9172325
32: -51.2251053, -16.1040516, -51.2251053, -16.1040516, -27.8580780, 27.8581314
33: -69.0680237, -12.0135298, -69.0680237, -12.0135298, -49.9538116, 49.9538574
34: -63.1368065, -21.4582806, -63.1368065, -21.4582806, -29.6437988, 29.6438522
35: -42.8285522, -0.5206113, -42.8285522, -0.5206113, -34.4654999, 34.4662628
36: -42.2597694, 2.7720976, -42.2597694, 2.7720976, -36.3751221, 36.3757172
37: -75.2216797, -18.9945145, -75.2216797, -18.9945145, -41.9191132, 41.9207153
38: -52.3679466, 2.0344648, -52.3679466, 2.0344648, -47.3311310, 47.3327026
39: -72.2596970, -13.6542854, -72.2596970, -13.6542854, -54.5074310, 54.5070953
40: -76.4585266, -36.7105141, -76.4585266, -36.7105141, -28.9931183, 28.9925232
41: -52.0074844, -11.1838779, -52.0074844, -11.1838779, -29.2403183, 29.2421875
42: -47.7828827, -16.1841316, -47.7828827, -16.1841316, -24.5300598, 24.5301247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=200, inp2_unstable=200, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1424
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9969975, upper bound: 15.0103613
time: 29.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0447218, upper bound: 14.9612361
time: 17.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 49.89 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 49.89
Output dim: 15, lower bound: -14.9612361, upper bound: 15.0447218
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 49.89
Output dim: 15, lower bound: -15.0103613, upper bound: 14.9969975
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 49.89
Output dim: 15, lower bound: -14.9969975, upper bound: 15.0103613
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 49.89
Output dim: 15, lower bound: -15.0447218, upper bound: 14.9612361

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.8484058, 27.0876694, -5.8484058, 27.0876694, -32.9360733, 32.9360733
1: 0.0080841, 22.9084625, 0.0080841, 22.9084625, -20.6868668, 20.6870270
2: -0.9063466, 25.1065483, -0.9063466, 25.1065483, -23.6731033, 23.6742630
3: -12.6358604, 15.9553623, -12.6358604, 15.9553623, -23.3050003, 23.3085938
4: -6.2847629, 20.8345890, -6.2847629, 20.8345890, -23.0598679, 23.0627823
5: -11.9136658, 20.9306450, -11.9136658, 20.9306450, -30.6637573, 30.6648254
6: -66.2194138, -30.6795292, -66.2194138, -30.6795292, -24.8202057, 24.8121147
7: -17.8853855, 14.9396420, -17.8853855, 14.9396420, -27.4222946, 27.4233170
8: -19.3295956, 10.2857838, -19.3295956, 10.2857838, -24.6263657, 24.6358757
9: -7.1940708, 22.8957558, -7.1940708, 22.8957558, -29.1298828, 29.1293411
10: -31.6227436, 12.2977619, -31.6227436, 12.2977619, -39.1621704, 39.1625824
11: -24.7147980, 0.1991603, -24.7147980, 0.1991603, -23.4446106, 23.4435463
12: -49.0830994, -8.5243492, -49.0830994, -8.5243492, -34.1215973, 34.1178970
13: -28.5756912, 19.1810932, -28.5756912, 19.1810932, -47.7567825, 47.7567825
14: -41.8362350, 8.8776398, -41.8362350, 8.8776398, -44.0437927, 44.0575104
15: -1.2438402, 28.1130295, -1.2438402, 28.1130295, -24.6908569, 24.7011108
16: -26.6712437, 10.5420656, -26.6712437, 10.5420656, -34.7429810, 34.7388992
17: -39.8865776, 4.8747683, -39.8865776, 4.8747683, -36.0362854, 36.0451431
18: -9.7624674, 26.5586090, -9.7624674, 26.5586090, -36.3210754, 36.3210754
19: -17.4268913, 9.3083858, -17.4268913, 9.3083858, -26.7352772, 26.7352772
20: -25.4231682, 3.7468839, -25.4231682, 3.7468839, -28.8565292, 28.8551331
21: -22.0199394, 8.0764503, -22.0199394, 8.0764503, -30.0963898, 30.0963898
22: -10.4554958, 21.3130703, -10.4554958, 21.3130703, -29.8040161, 29.8069382
23: -10.3478737, 18.1226082, -10.3478737, 18.1226082, -28.2311172, 28.2303925
24: -6.4852304, 22.3751621, -6.4852304, 22.3751621, -28.2761536, 28.2779541
25: -9.5180445, 22.8397675, -9.5180445, 22.8397675, -32.2773285, 32.2822266
26: -23.2106190, 24.4033546, -23.2106190, 24.4033546, -47.0097961, 47.0122375
27: -15.9029713, 17.4331760, -15.9029713, 17.4331760, -33.3361473, 33.3361473
28: -13.9861851, 19.3545685, -13.9861851, 19.3545685, -31.5548553, 31.5527954
29: -10.9152431, 15.4316082, -10.9152431, 15.4316082, -23.1451263, 23.1455536
30: -30.3633671, -0.1006818, -30.3633671, -0.1006818, -27.8794098, 27.8794250
31: -18.1789322, 10.7383003, -18.1789322, 10.7383003, -28.9172325, 28.9172325
32: -51.2251053, -16.1040516, -51.2251053, -16.1040516, -27.8338165, 27.8292084
33: -69.0680237, -12.0135298, -69.0680237, -12.0135298, -49.9447937, 49.9431915
34: -63.1368065, -21.4582806, -63.1368065, -21.4582806, -29.6268311, 29.6234741
35: -42.8285522, -0.5206113, -42.8285522, -0.5206113, -34.4442139, 34.4400177
36: -42.2597694, 2.7720976, -42.2597694, 2.7720976, -36.3563995, 36.3525391
37: -75.2216797, -18.9945145, -75.2216797, -18.9945145, -41.9049225, 41.9017105
38: -52.3679466, 2.0344648, -52.3679466, 2.0344648, -47.3071136, 47.3005371
39: -72.2596970, -13.6542854, -72.2596970, -13.6542854, -54.4970245, 54.4969940
40: -76.4585266, -36.7105141, -76.4585266, -36.7105141, -28.9928513, 28.9928894
41: -52.0074844, -11.1838779, -52.0074844, -11.1838779, -29.2042160, 29.1944733
42: -47.7828827, -16.1841316, -47.7828827, -16.1841316, -24.5279999, 24.5276947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=200, inp2_unstable=200, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1424
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9602619, upper bound: 15.0127543
time: 22.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9292292, upper bound: 15.0437555
time: 35.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.8484058, 27.0876694, -5.8484058, 27.0876694, -32.9360733, 32.9360733
1: 0.0080841, 22.9084625, 0.0080841, 22.9084625, -20.6870270, 20.6868591
2: -0.9063466, 25.1065483, -0.9063466, 25.1065483, -23.6742630, 23.6731033
3: -12.6358604, 15.9553623, -12.6358604, 15.9553623, -23.3085938, 23.3050003
4: -6.2847629, 20.8345890, -6.2847629, 20.8345890, -23.0627823, 23.0598679
5: -11.9136658, 20.9306450, -11.9136658, 20.9306450, -30.6648254, 30.6637650
6: -66.2194138, -30.6795292, -66.2194138, -30.6795292, -24.8121185, 24.8202095
7: -17.8853855, 14.9396420, -17.8853855, 14.9396420, -27.4233246, 27.4222946
8: -19.3295956, 10.2857838, -19.3295956, 10.2857838, -24.6358719, 24.6263618
9: -7.1940708, 22.8957558, -7.1940708, 22.8957558, -29.1293488, 29.1298828
10: -31.6227436, 12.2977619, -31.6227436, 12.2977619, -39.1625671, 39.1621704
11: -24.7147980, 0.1991603, -24.7147980, 0.1991603, -23.4435425, 23.4446106
12: -49.0830994, -8.5243492, -49.0830994, -8.5243492, -34.1179047, 34.1215973
13: -28.5756912, 19.1810932, -28.5756912, 19.1810932, -47.7567825, 47.7567825
14: -41.8362350, 8.8776398, -41.8362350, 8.8776398, -44.0575104, 44.0437927
15: -1.2438402, 28.1130295, -1.2438402, 28.1130295, -24.7011108, 24.6908569
16: -26.6712437, 10.5420656, -26.6712437, 10.5420656, -34.7388916, 34.7429886
17: -39.8865776, 4.8747683, -39.8865776, 4.8747683, -36.0451355, 36.0362854
18: -9.7624674, 26.5586090, -9.7624674, 26.5586090, -36.3210754, 36.3210754
19: -17.4268913, 9.3083858, -17.4268913, 9.3083858, -26.7352772, 26.7352772
20: -25.4231682, 3.7468839, -25.4231682, 3.7468839, -28.8551407, 28.8565369
21: -22.0199394, 8.0764503, -22.0199394, 8.0764503, -30.0963898, 30.0963898
22: -10.4554958, 21.3130703, -10.4554958, 21.3130703, -29.8069458, 29.8040161
23: -10.3478737, 18.1226082, -10.3478737, 18.1226082, -28.2303848, 28.2311096
24: -6.4852304, 22.3751621, -6.4852304, 22.3751621, -28.2779541, 28.2761536
25: -9.5180445, 22.8397675, -9.5180445, 22.8397675, -32.2822113, 32.2773209
26: -23.2106190, 24.4033546, -23.2106190, 24.4033546, -47.0122375, 47.0097961
27: -15.9029713, 17.4331760, -15.9029713, 17.4331760, -33.3361473, 33.3361473
28: -13.9861851, 19.3545685, -13.9861851, 19.3545685, -31.5527954, 31.5548553
29: -10.9152431, 15.4316082, -10.9152431, 15.4316082, -23.1455536, 23.1451263
30: -30.3633671, -0.1006818, -30.3633671, -0.1006818, -27.8794250, 27.8794022
31: -18.1789322, 10.7383003, -18.1789322, 10.7383003, -28.9172325, 28.9172325
32: -51.2251053, -16.1040516, -51.2251053, -16.1040516, -27.8292084, 27.8338165
33: -69.0680237, -12.0135298, -69.0680237, -12.0135298, -49.9431763, 49.9448090
34: -63.1368065, -21.4582806, -63.1368065, -21.4582806, -29.6234741, 29.6268311
35: -42.8285522, -0.5206113, -42.8285522, -0.5206113, -34.4400177, 34.4442062
36: -42.2597694, 2.7720976, -42.2597694, 2.7720976, -36.3525391, 36.3563995
37: -75.2216797, -18.9945145, -75.2216797, -18.9945145, -41.9017181, 41.9049225
38: -52.3679466, 2.0344648, -52.3679466, 2.0344648, -47.3005371, 47.3071136
39: -72.2596970, -13.6542854, -72.2596970, -13.6542854, -54.4969940, 54.4970398
40: -76.4585266, -36.7105141, -76.4585266, -36.7105141, -28.9928894, 28.9928513
41: -52.0074844, -11.1838779, -52.0074844, -11.1838779, -29.1944733, 29.2042160
42: -47.7828827, -16.1841316, -47.7828827, -16.1841316, -24.5276947, 24.5279999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=200, inp2_unstable=200, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1424
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0437555, upper bound: 14.9292292
time: 20.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0127543, upper bound: 14.9602619
time: 20.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 44.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 44.20
Output dim: 15, lower bound: -14.9602619, upper bound: 15.0127543
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 44.20
Output dim: 15, lower bound: -14.9292292, upper bound: 15.0437555
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 44.20
Output dim: 15, lower bound: -15.0437555, upper bound: 14.9292292
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 44.20
Output dim: 15, lower bound: -15.0127543, upper bound: 14.9602619

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.8484058, 27.0876694, -5.8484058, 27.0876694, -32.9360733, 32.9360733
1: 0.0080841, 22.9084625, 0.0080841, 22.9084625, -20.6868286, 20.6867294
2: -0.9063466, 25.1065483, -0.9063466, 25.1065483, -23.6691742, 23.6722565
3: -12.6358604, 15.9553623, -12.6358604, 15.9553623, -23.3049850, 23.3085861
4: -6.2847629, 20.8345890, -6.2847629, 20.8345890, -23.0538330, 23.0597000
5: -11.9136658, 20.9306450, -11.9136658, 20.9306450, -30.6595154, 30.6623764
6: -66.2194138, -30.6795292, -66.2194138, -30.6795292, -24.8086548, 24.7980614
7: -17.8853855, 14.9396420, -17.8853855, 14.9396420, -27.4220734, 27.4232864
8: -19.3295956, 10.2857838, -19.3295956, 10.2857838, -24.6202621, 24.6327705
9: -7.1940708, 22.8957558, -7.1940708, 22.8957558, -29.1240616, 29.1244888
10: -31.6227436, 12.2977619, -31.6227436, 12.2977619, -39.1576462, 39.1590042
11: -24.7147980, 0.1991603, -24.7147980, 0.1991603, -23.4411163, 23.4387512
12: -49.0830994, -8.5243492, -49.0830994, -8.5243492, -34.1165924, 34.1072083
13: -28.5756912, 19.1810932, -28.5756912, 19.1810932, -47.7567825, 47.7567825
14: -41.8362350, 8.8776398, -41.8362350, 8.8776398, -44.0390320, 44.0534515
15: -1.2438402, 28.1130295, -1.2438402, 28.1130295, -24.6810608, 24.6932297
16: -26.6712437, 10.5420656, -26.6712437, 10.5420656, -34.7418442, 34.7377472
17: -39.8865776, 4.8747683, -39.8865776, 4.8747683, -36.0358124, 36.0438995
18: -9.7624674, 26.5586090, -9.7624674, 26.5586090, -36.3210754, 36.3210754
19: -17.4268913, 9.3083858, -17.4268913, 9.3083858, -26.7352772, 26.7352772
20: -25.4231682, 3.7468839, -25.4231682, 3.7468839, -28.8533859, 28.8524704
21: -22.0199394, 8.0764503, -22.0199394, 8.0764503, -30.0963898, 30.0963898
22: -10.4554958, 21.3130703, -10.4554958, 21.3130703, -29.8034897, 29.8089371
23: -10.3478737, 18.1226082, -10.3478737, 18.1226082, -28.2306824, 28.2305832
24: -6.4852304, 22.3751621, -6.4852304, 22.3751621, -28.2753983, 28.2782364
25: -9.5180445, 22.8397675, -9.5180445, 22.8397675, -32.2764282, 32.2839127
26: -23.2106190, 24.4033546, -23.2106190, 24.4033546, -47.0096283, 47.0120544
27: -15.9029713, 17.4331760, -15.9029713, 17.4331760, -33.3361473, 33.3361473
28: -13.9861851, 19.3545685, -13.9861851, 19.3545685, -31.5532990, 31.5499802
29: -10.9152431, 15.4316082, -10.9152431, 15.4316082, -23.1449966, 23.1452637
30: -30.3633671, -0.1006818, -30.3633671, -0.1006818, -27.8739395, 27.8717728
31: -18.1789322, 10.7383003, -18.1789322, 10.7383003, -28.9172325, 28.9172325
32: -51.2251053, -16.1040516, -51.2251053, -16.1040516, -27.8274384, 27.8211517
33: -69.0680237, -12.0135298, -69.0680237, -12.0135298, -49.9440155, 49.9424591
34: -63.1368065, -21.4582806, -63.1368065, -21.4582806, -29.6186523, 29.6079636
35: -42.8285522, -0.5206113, -42.8285522, -0.5206113, -34.4407196, 34.4334869
36: -42.2597694, 2.7720976, -42.2597694, 2.7720976, -36.3539429, 36.3478394
37: -75.2216797, -18.9945145, -75.2216797, -18.9945145, -41.9048920, 41.9016190
38: -52.3679466, 2.0344648, -52.3679466, 2.0344648, -47.3018494, 47.2897949
39: -72.2596970, -13.6542854, -72.2596970, -13.6542854, -54.4911804, 54.4944458
40: -76.4585266, -36.7105141, -76.4585266, -36.7105141, -28.9901047, 28.9893951
41: -52.0074844, -11.1838779, -52.0074844, -11.1838779, -29.1988144, 29.1858063
42: -47.7828827, -16.1841316, -47.7828827, -16.1841316, -24.5211792, 24.5198135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=200, inp2_unstable=200, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1424
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 706

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9079389, upper bound: 15.0007585
time: 29.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9052422, upper bound: 15.0118949
time: 29.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.8484058, 27.0876694, -5.8484058, 27.0876694, -32.9360733, 32.9360733
1: 0.0080841, 22.9084625, 0.0080841, 22.9084625, -20.6867294, 20.6868286
2: -0.9063466, 25.1065483, -0.9063466, 25.1065483, -23.6722565, 23.6691742
3: -12.6358604, 15.9553623, -12.6358604, 15.9553623, -23.3085861, 23.3049850
4: -6.2847629, 20.8345890, -6.2847629, 20.8345890, -23.0597000, 23.0538330
5: -11.9136658, 20.9306450, -11.9136658, 20.9306450, -30.6623688, 30.6595078
6: -66.2194138, -30.6795292, -66.2194138, -30.6795292, -24.7980652, 24.8086510
7: -17.8853855, 14.9396420, -17.8853855, 14.9396420, -27.4232864, 27.4220810
8: -19.3295956, 10.2857838, -19.3295956, 10.2857838, -24.6327744, 24.6202621
9: -7.1940708, 22.8957558, -7.1940708, 22.8957558, -29.1244888, 29.1240616
10: -31.6227436, 12.2977619, -31.6227436, 12.2977619, -39.1590042, 39.1576462
11: -24.7147980, 0.1991603, -24.7147980, 0.1991603, -23.4387512, 23.4411125
12: -49.0830994, -8.5243492, -49.0830994, -8.5243492, -34.1072083, 34.1165924
13: -28.5756912, 19.1810932, -28.5756912, 19.1810932, -47.7567825, 47.7567825
14: -41.8362350, 8.8776398, -41.8362350, 8.8776398, -44.0534515, 44.0390320
15: -1.2438402, 28.1130295, -1.2438402, 28.1130295, -24.6932297, 24.6810608
16: -26.6712437, 10.5420656, -26.6712437, 10.5420656, -34.7377396, 34.7418442
17: -39.8865776, 4.8747683, -39.8865776, 4.8747683, -36.0438995, 36.0358124
18: -9.7624674, 26.5586090, -9.7624674, 26.5586090, -36.3210754, 36.3210754
19: -17.4268913, 9.3083858, -17.4268913, 9.3083858, -26.7352772, 26.7352772
20: -25.4231682, 3.7468839, -25.4231682, 3.7468839, -28.8524704, 28.8533859
21: -22.0199394, 8.0764503, -22.0199394, 8.0764503, -30.0963898, 30.0963898
22: -10.4554958, 21.3130703, -10.4554958, 21.3130703, -29.8089371, 29.8034897
23: -10.3478737, 18.1226082, -10.3478737, 18.1226082, -28.2305908, 28.2306747
24: -6.4852304, 22.3751621, -6.4852304, 22.3751621, -28.2782364, 28.2753983
25: -9.5180445, 22.8397675, -9.5180445, 22.8397675, -32.2839050, 32.2764359
26: -23.2106190, 24.4033546, -23.2106190, 24.4033546, -47.0120544, 47.0096283
27: -15.9029713, 17.4331760, -15.9029713, 17.4331760, -33.3361473, 33.3361473
28: -13.9861851, 19.3545685, -13.9861851, 19.3545685, -31.5499878, 31.5532990
29: -10.9152431, 15.4316082, -10.9152431, 15.4316082, -23.1452637, 23.1449966
30: -30.3633671, -0.1006818, -30.3633671, -0.1006818, -27.8717728, 27.8739395
31: -18.1789322, 10.7383003, -18.1789322, 10.7383003, -28.9172325, 28.9172325
32: -51.2251053, -16.1040516, -51.2251053, -16.1040516, -27.8211517, 27.8274384
33: -69.0680237, -12.0135298, -69.0680237, -12.0135298, -49.9424591, 49.9440155
34: -63.1368065, -21.4582806, -63.1368065, -21.4582806, -29.6079636, 29.6186523
35: -42.8285522, -0.5206113, -42.8285522, -0.5206113, -34.4334869, 34.4407196
36: -42.2597694, 2.7720976, -42.2597694, 2.7720976, -36.3478394, 36.3539429
37: -75.2216797, -18.9945145, -75.2216797, -18.9945145, -41.9016266, 41.9048920
38: -52.3679466, 2.0344648, -52.3679466, 2.0344648, -47.2897949, 47.3018494
39: -72.2596970, -13.6542854, -72.2596970, -13.6542854, -54.4944458, 54.4911652
40: -76.4585266, -36.7105141, -76.4585266, -36.7105141, -28.9893951, 28.9901047
41: -52.0074844, -11.1838779, -52.0074844, -11.1838779, -29.1858063, 29.1988144
42: -47.7828827, -16.1841316, -47.7828827, -16.1841316, -24.5198135, 24.5211830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=200, inp2_unstable=200, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1424
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1452

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 706

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0118949, upper bound: 14.9052422
time: 22.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0007585, upper bound: 14.9079389
time: 19.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 44.81 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 44.81
Output dim: 15, lower bound: -14.9079389, upper bound: 15.0007585
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 44.81
Output dim: 15, lower bound: -14.9052422, upper bound: 15.0118949
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 44.81
Output dim: 15, lower bound: -15.0118949, upper bound: 14.9052422
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 44.81
Output dim: 15, lower bound: -15.0007585, upper bound: 14.9079389

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 38.30 + 362.46 = 400.76 seconds
