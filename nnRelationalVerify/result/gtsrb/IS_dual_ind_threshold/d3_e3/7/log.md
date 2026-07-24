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
execution time: IAR + RelationalAnalysis = 2.94 + 34.20 = 37.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 15, lower bound: -15.0476571, upper bound: 15.0476571

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1731

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1714

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0249267, upper bound: 15.0454064
time: 26.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0456932, upper bound: 15.0456933
time: 23.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 49.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 49.79
Output dim: 15, lower bound: -15.0249267, upper bound: 15.0454064
IS_A2, status: Status.UNKNOWN, split count: 1, time: 49.79
Output dim: 15, lower bound: -15.0456932, upper bound: 15.0456933

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.7268591, 27.0472755, -5.7912188, 27.0832825, -32.8101425, 32.8384933
1: 0.0655348, 22.8891525, 0.0339580, 22.9049473, -20.6307526, 20.6415176
2: -0.8340936, 25.0678158, -0.8711736, 25.1030579, -23.6042633, 23.6050568
3: -12.5735703, 15.9277525, -12.6054173, 15.9515638, -23.2590027, 23.3175049
4: -6.2164569, 20.8252449, -6.2528276, 20.8322029, -23.0019684, 23.0252533
5: -11.8585052, 20.9043827, -11.8872547, 20.9257851, -30.6004715, 30.6169662
6: -66.1944351, -30.7489815, -66.2151794, -30.7128029, -24.8031158, 24.7944450
7: -17.8345566, 14.9251509, -17.8635063, 14.9365997, -27.3649826, 27.4058151
8: -19.2299557, 10.2374954, -19.2799187, 10.2800379, -24.5694656, 24.6121368
9: -7.1584134, 22.8722534, -7.1800280, 22.8853645, -29.0759506, 29.0662231
10: -31.5579758, 12.2327423, -31.6095390, 12.2683992, -39.0576019, 38.9457397
11: -24.6662197, 0.1415329, -24.7098522, 0.1715841, -23.3871994, 23.4124069
12: -49.0678329, -8.5931444, -49.0782623, -8.5528660, -34.0918579, 34.0573883
13: -28.5056114, 19.1050091, -28.5420227, 19.1703835, -47.6759949, 47.6470337
14: -41.7604523, 8.8354816, -41.8018608, 8.8730793, -44.0331726, 43.9701462
15: -1.1468325, 28.0589848, -1.1986580, 28.1064682, -24.6509781, 24.6616707
16: -26.6347389, 10.5249147, -26.6633358, 10.5342464, -34.6639252, 34.6459503
17: -39.7687645, 4.8037810, -39.8338776, 4.8699903, -35.8643570, 35.9565125
18: -9.6868601, 26.5094070, -9.7555428, 26.5344849, -36.2213440, 36.2649498
19: -17.3746109, 9.2502346, -17.4217682, 9.2788334, -26.6534443, 26.6720028
20: -25.3543949, 3.6710136, -25.4169769, 3.7091599, -28.7637024, 28.7961731
21: -21.9355392, 7.9682498, -22.0137329, 8.0223160, -29.9578552, 29.9819832
22: -10.4233437, 21.2861309, -10.4501419, 21.3004837, -29.7249298, 29.7430344
23: -10.3069220, 18.0772343, -10.3424253, 18.1015644, -28.1698303, 28.1864014
24: -6.4303050, 22.3343391, -6.4776449, 22.3548508, -28.1953278, 28.2173538
25: -9.4695807, 22.7849140, -9.5105915, 22.8133450, -32.1679230, 32.1850052
26: -23.1607590, 24.3652306, -23.2020569, 24.3845978, -46.8949280, 46.9250031
27: -15.8616648, 17.3910332, -15.8947840, 17.4129219, -33.2745857, 33.2858162
28: -13.9436598, 19.3179455, -13.9793034, 19.3372459, -31.4835281, 31.5198669
29: -10.8957520, 15.4004335, -10.9089508, 15.4190798, -23.1008072, 23.1056747
30: -30.2937889, -0.1733446, -30.3584633, -0.1344740, -27.7800064, 27.8223801
31: -18.1081886, 10.6505709, -18.1707745, 10.6944370, -28.8026257, 28.8213463
32: -51.1979141, -16.1541138, -51.2185135, -16.1242886, -27.8042450, 27.8033829
33: -69.0317383, -12.0590935, -69.0586548, -12.0330429, -49.8894501, 49.8985748
34: -63.0946579, -21.4896088, -63.1288834, -21.4708977, -29.6154175, 29.5994644
35: -42.8092613, -0.5426161, -42.8229485, -0.5284317, -34.4136963, 34.4340439
36: -42.2425919, 2.7499189, -42.2532120, 2.7636528, -36.3053741, 36.3448944
37: -75.1917572, -19.0151119, -75.2118988, -19.0016937, -41.8624878, 41.8947754
38: -52.3128166, 1.9815550, -52.3556404, 2.0126252, -47.2472534, 47.2652283
39: -72.2254868, -13.6942062, -72.2502747, -13.6688957, -54.4523621, 54.4559174
40: -76.4180298, -36.7368317, -76.4444656, -36.7165298, -28.9508438, 28.9557877
41: -51.9938469, -11.2161427, -52.0032845, -11.1967964, -29.1549911, 29.2025375
42: -47.7692108, -16.2155800, -47.7794800, -16.1965790, -24.5071030, 24.5254745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=199, inp2_unstable=200, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1729

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9982275, upper bound: 14.9643851
time: 21.19 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0235870, upper bound: 15.0439875
time: 18.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.8411884, 27.0866699, -5.8448744, 27.0872040, -32.9283905, 32.9315453
1: 0.0145087, 22.9077053, 0.0113032, 22.9081039, -20.6890869, 20.6982422
2: -0.9019675, 25.1053905, -0.9040477, 25.1059933, -23.6638336, 23.6771545
3: -12.6316452, 15.9546824, -12.6337337, 15.9550600, -23.3194656, 23.3254166
4: -6.2785640, 20.8337059, -6.2816982, 20.8341675, -23.0621033, 23.0667725
5: -11.9088821, 20.9296970, -11.9113531, 20.9301929, -30.6580505, 30.6623611
6: -66.2181702, -30.6880035, -66.2187958, -30.6841011, -24.8667221, 24.8499298
7: -17.8805065, 14.9390278, -17.8830738, 14.9393578, -27.4210815, 27.4284134
8: -19.3224640, 10.2847233, -19.3260174, 10.2852345, -24.6462708, 24.6698151
9: -7.1922393, 22.8914967, -7.1930947, 22.8936100, -29.1150055, 29.1228790
10: -31.6198578, 12.2907286, -31.6213646, 12.2942629, -39.1528854, 39.1243591
11: -24.7124348, 0.1959269, -24.7136841, 0.1973839, -23.4500580, 23.4371643
12: -49.0821609, -8.5313997, -49.0826340, -8.5277843, -34.1328354, 34.1245422
13: -28.5710907, 19.1782494, -28.5734901, 19.1797295, -47.7508202, 47.7517395
14: -41.8277092, 8.8765888, -41.8317337, 8.8771324, -44.1020203, 44.1065979
15: -1.2384391, 28.1102238, -1.2411714, 28.1117001, -24.7166595, 24.7451248
16: -26.6686192, 10.5371647, -26.6699829, 10.5396833, -34.7092133, 34.7099533
17: -39.8780098, 4.8739786, -39.8824730, 4.8744168, -36.0070496, 36.0814133
18: -9.7613659, 26.5559044, -9.7619820, 26.5572472, -36.3186111, 36.3178864
19: -17.4253426, 9.3052063, -17.4261017, 9.3064117, -26.7317543, 26.7313080
20: -25.4212265, 3.7428660, -25.4222527, 3.7446926, -28.8663330, 28.8534622
21: -22.0169926, 8.0700712, -22.0185299, 8.0733728, -30.0903664, 30.0886002
22: -10.4538631, 21.3080673, -10.4547215, 21.3106556, -29.7923279, 29.7834091
23: -10.3463306, 18.1174412, -10.3471279, 18.1201019, -28.2262115, 28.2178955
24: -6.4833527, 22.3734913, -6.4843192, 22.3743362, -28.2686157, 28.2649307
25: -9.5154305, 22.8352032, -9.5167656, 22.8375225, -32.2539520, 32.2510986
26: -23.2088528, 24.3965263, -23.2097855, 24.4000454, -47.0121307, 46.9824982
27: -15.9014587, 17.4268818, -15.9022369, 17.4301548, -33.3316116, 33.3291168
28: -13.9843283, 19.3483849, -13.9852924, 19.3515015, -31.5548401, 31.5472260
29: -10.9109392, 15.4293680, -10.9131060, 15.4305553, -23.1370621, 23.1401672
30: -30.3609409, -0.1039269, -30.3622055, -0.1022573, -27.8786697, 27.8571167
31: -18.1760845, 10.7349606, -18.1775398, 10.7366848, -28.9127693, 28.9125004
32: -51.2237701, -16.1116447, -51.2244606, -16.1077328, -27.8542862, 27.8418427
33: -69.0653915, -12.0228920, -69.0667648, -12.0180292, -49.9470062, 49.9393768
34: -63.1352654, -21.4632587, -63.1360588, -21.4607773, -29.6407852, 29.6309204
35: -42.8273087, -0.5220582, -42.8279190, -0.5212841, -34.4651794, 34.4650116
36: -42.2560883, 2.7690291, -42.2579727, 2.7705534, -36.3741608, 36.3694153
37: -75.2180557, -18.9957829, -75.2199020, -18.9951286, -41.9168091, 41.9189529
38: -52.3660088, 2.0271788, -52.3669891, 2.0304937, -47.3377991, 47.3231812
39: -72.2575989, -13.6641693, -72.2587128, -13.6591091, -54.5015259, 54.4937134
40: -76.4552231, -36.7120514, -76.4569168, -36.7112579, -28.9849243, 28.9897461
41: -52.0065689, -11.1893721, -52.0070343, -11.1865320, -29.2401657, 29.2444458
42: -47.7815018, -16.1891422, -47.7821922, -16.1866341, -24.5264359, 24.5244522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=199, inp2_unstable=200, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1729

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0249191, upper bound: 14.9680037
time: 18.94 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0442341, upper bound: 15.0442340
time: 27.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 49.14 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 49.14
Output dim: 15, lower bound: -14.9982275, upper bound: 14.9643851
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 49.14
Output dim: 15, lower bound: -15.0235870, upper bound: 15.0439875
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 49.14
Output dim: 15, lower bound: -15.0249191, upper bound: 14.9680037
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 49.14
Output dim: 15, lower bound: -15.0442341, upper bound: 15.0442340

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.7260094, 27.0471954, -5.7883368, 27.0829391, -32.8089485, 32.8355331
1: 0.0660558, 22.8891277, 0.0359232, 22.9046745, -20.6302032, 20.6327515
2: -0.8336477, 25.0677414, -0.8692412, 25.1027870, -23.6035461, 23.5988159
3: -12.5731735, 15.9276810, -12.6038418, 15.9513636, -23.2585678, 23.3034821
4: -6.2160144, 20.8251820, -6.2507873, 20.8318939, -23.0014191, 23.0181160
5: -11.8579741, 20.9042702, -11.8850555, 20.9254780, -30.5999756, 30.6145935
6: -66.1944275, -30.7493782, -66.2149963, -30.7147560, -24.7528687, 24.7940483
7: -17.8340530, 14.9251194, -17.8617477, 14.9363279, -27.3643494, 27.4022446
8: -19.2291317, 10.2373533, -19.2763596, 10.2794819, -24.5680847, 24.5769386
9: -7.1579990, 22.8722744, -7.1779943, 22.8852348, -29.0752792, 29.0637589
10: -31.5571747, 12.2326107, -31.6059685, 12.2677631, -39.0606766, 38.9422455
11: -24.6659889, 0.1412640, -24.7088051, 0.1704755, -23.3768005, 23.4112282
12: -49.0676956, -8.5936050, -49.0777054, -8.5551634, -34.0787964, 34.0563354
13: -28.5048256, 19.1048393, -28.5386124, 19.1697235, -47.6745491, 47.6434517
14: -41.7595062, 8.8354549, -41.7972260, 8.8728313, -44.0323029, 43.9112473
15: -1.1461673, 28.0588627, -1.1955557, 28.1061211, -24.6501846, 24.6140289
16: -26.6340332, 10.5247707, -26.6601124, 10.5338297, -34.6869354, 34.6420517
17: -39.7680321, 4.8036556, -39.8306770, 4.8696165, -35.8635788, 35.9180984
18: -9.6867619, 26.5091858, -9.7549467, 26.5339298, -36.2206917, 36.2641335
19: -17.3745079, 9.2499094, -17.4212780, 9.2776165, -26.6521244, 26.6711884
20: -25.3542747, 3.6706851, -25.4164009, 3.7077665, -28.7519531, 28.7955933
21: -21.9354153, 7.9676380, -22.0128708, 8.0209532, -29.9563675, 29.9805088
22: -10.4232082, 21.2855110, -10.4494572, 21.2978840, -29.7221298, 29.7582016
23: -10.3068237, 18.0768280, -10.3419275, 18.0996704, -28.1680222, 28.1877060
24: -6.4302149, 22.3340569, -6.4771919, 22.3538799, -28.1930084, 28.2258301
25: -9.4694242, 22.7844982, -9.5099859, 22.8118610, -32.1641006, 32.2103729
26: -23.1606331, 24.3645554, -23.2014275, 24.3815861, -46.8917542, 46.9391479
27: -15.8615685, 17.3903942, -15.8943100, 17.4100666, -33.2716370, 33.2847061
28: -13.9435682, 19.3175926, -13.9787664, 19.3353653, -31.4761505, 31.5185165
29: -10.8956194, 15.4000940, -10.9083986, 15.4175406, -23.0984039, 23.1084595
30: -30.2936668, -0.1739473, -30.3578777, -0.1373398, -27.7751007, 27.8214264
31: -18.1080246, 10.6502151, -18.1700172, 10.6933231, -28.8013477, 28.8202324
32: -51.1978378, -16.1544704, -51.2181435, -16.1261196, -27.7784500, 27.8027420
33: -69.0315704, -12.0597878, -69.0580597, -12.0362339, -49.8771210, 49.8971710
34: -63.0945778, -21.4900818, -63.1283646, -21.4729919, -29.5965118, 29.5985641
35: -42.8091621, -0.5431604, -42.8225098, -0.5309715, -34.3894348, 34.4330673
36: -42.2425156, 2.7493997, -42.2529602, 2.7607458, -36.2830963, 36.3439484
37: -75.1916580, -19.0155697, -75.2115326, -19.0037594, -41.8440552, 41.8939667
38: -52.3126793, 1.9805493, -52.3548584, 2.0080929, -47.2176208, 47.2634583
39: -72.2253265, -13.6949921, -72.2496643, -13.6726379, -54.4386139, 54.4545898
40: -76.4179306, -36.7373581, -76.4440384, -36.7189827, -28.9498596, 28.9548721
41: -51.9937935, -11.2165089, -52.0031700, -11.1988697, -29.1131516, 29.2019806
42: -47.7691650, -16.2159882, -47.7792320, -16.1982346, -24.5032043, 24.5248604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=199, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9858748, upper bound: 15.0390840
time: 21.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0215078, upper bound: 15.0424561
time: 21.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.7461252, 27.0803127, -5.6343222, 27.0558853, -32.8020096, 32.7146339
1: 0.0683801, 22.9027443, 0.1309724, 22.8845100, -20.6127014, 20.5730743
2: -0.8503182, 25.1003971, -0.7899482, 25.0813217, -23.5879669, 23.5574417
3: -12.5723619, 15.9504385, -12.5035028, 15.9307919, -23.2346725, 23.1899872
4: -6.2211251, 20.8286209, -6.1531701, 20.8201237, -22.9904404, 22.9320526
5: -11.8551006, 20.9241371, -11.7920790, 20.9044666, -30.5735779, 30.5306091
6: -66.2138519, -30.7541389, -66.1915512, -30.8323059, -24.7072220, 24.7526588
7: -17.8325005, 14.9356661, -17.7765903, 14.9234676, -27.3508301, 27.3072433
8: -19.2299576, 10.2763367, -19.1220131, 10.2445755, -24.5147476, 24.4608231
9: -7.1442685, 22.8864250, -7.0852685, 22.8743687, -29.0401917, 29.0026398
10: -31.5858955, 12.2772255, -31.5446167, 12.2591019, -39.0737610, 39.0233688
11: -24.6997147, 0.1559880, -24.6722412, 0.1086655, -23.3491821, 23.3571167
12: -49.0759964, -8.5895824, -49.0606346, -8.6608334, -33.9949875, 34.0418472
13: -28.5504494, 19.1626472, -28.5266113, 19.1432590, -47.6937103, 47.6892586
14: -41.7553215, 8.8704128, -41.6693192, 8.8415432, -43.9793701, 43.9258118
15: -1.1624107, 28.1003742, -1.0697832, 28.0681229, -24.5937424, 24.5589333
16: -26.6430626, 10.5276709, -26.6093178, 10.5161171, -34.6234818, 34.6138077
17: -39.8232613, 4.8663378, -39.7604942, 4.8419232, -35.9174805, 35.9471436
18: -9.7527151, 26.5199680, -9.7212381, 26.4777527, -36.2304688, 36.2412071
19: -17.4171867, 9.2529974, -17.3868065, 9.1914673, -26.6086540, 26.6398048
20: -25.4102764, 3.6927083, -25.3735390, 3.6332369, -28.7439194, 28.7561417
21: -22.0035801, 8.0061331, -21.9629555, 7.9317923, -29.9353714, 29.9690895
22: -10.4436140, 21.2820892, -10.4254551, 21.2523918, -29.7022247, 29.7027893
23: -10.3370161, 18.0886688, -10.3161974, 18.0568237, -28.1502151, 28.1544647
24: -6.4740772, 22.3404961, -6.4450703, 22.3009338, -28.1826401, 28.1853561
25: -9.5022306, 22.7983513, -9.4752054, 22.7552662, -32.1463318, 32.1509476
26: -23.1975079, 24.3581505, -23.1707840, 24.3144512, -46.8858643, 46.8746490
27: -15.8925428, 17.3958664, -15.8811350, 17.3620014, -33.2545433, 33.2770004
28: -13.9743252, 19.3114281, -13.9502068, 19.2703972, -31.4696198, 31.4803543
29: -10.9051847, 15.4136868, -10.9017448, 15.3940105, -23.0920715, 23.1074982
30: -30.3506413, -0.1377385, -30.3223228, -0.1769238, -27.7922974, 27.7811813
31: -18.1621513, 10.6721230, -18.1230259, 10.5987005, -28.7608528, 28.7951488
32: -51.2167206, -16.1554394, -51.1958351, -16.2090034, -27.7441025, 27.7689819
33: -69.0486603, -12.0863972, -69.0027161, -12.1605482, -49.7872772, 49.8110809
34: -63.1251755, -21.5010147, -63.0950394, -21.5454559, -29.5386047, 29.5498886
35: -42.8176003, -0.5703971, -42.7845993, -0.6292467, -34.3397827, 34.3689804
36: -42.2495918, 2.7084982, -42.2215843, 2.6346407, -36.2273254, 36.2704849
37: -75.2096176, -19.0371056, -75.1845703, -19.0870228, -41.8133392, 41.8379517
38: -52.3496628, 1.9378777, -52.2904701, 1.8294449, -47.1205292, 47.1589661
39: -72.2418289, -13.7329845, -72.1926575, -13.8139210, -54.3307953, 54.3590546
40: -76.4418716, -36.7299614, -76.4263687, -36.7526131, -28.9324722, 28.9417114
41: -52.0024109, -11.2340889, -51.9844971, -11.2884331, -29.1327438, 29.1732254
42: -47.7768021, -16.2106552, -47.7675171, -16.2373676, -24.4700851, 24.4883499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=199, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=144, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9751524, upper bound: 14.9599643
time: 27.54 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0236999, upper bound: 14.9663664
time: 24.23 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.8403492, 27.0866222, -5.8419876, 27.0868683, -32.9272156, 32.9286118
1: 0.0150430, 22.9076920, 0.0132477, 22.9078789, -20.6885223, 20.6894989
2: -0.9014933, 25.1053219, -0.9021306, 25.1056976, -23.6631165, 23.6708755
3: -12.6312885, 15.9545898, -12.6320915, 15.9547768, -23.3190231, 23.3117142
4: -6.2781029, 20.8336372, -6.2795897, 20.8339043, -23.0615845, 23.0596695
5: -11.9084091, 20.9296417, -11.9091377, 20.9298573, -30.6575775, 30.6599808
6: -66.2181396, -30.6884079, -66.2186737, -30.6860504, -24.8165512, 24.8495064
7: -17.8800793, 14.9389629, -17.8812408, 14.9391413, -27.4204330, 27.4248352
8: -19.3216400, 10.2845888, -19.3224640, 10.2847261, -24.6448746, 24.6346283
9: -7.1917925, 22.8914890, -7.1911802, 22.8934231, -29.1143341, 29.1204224
10: -31.6191063, 12.2906113, -31.6177788, 12.2936640, -39.1558609, 39.1202240
11: -24.7121964, 0.1956985, -24.7125950, 0.1962688, -23.4396820, 23.4359970
12: -49.0820236, -8.5319405, -49.0820961, -8.5300522, -34.1196899, 34.1235352
13: -28.5703316, 19.1781387, -28.5700722, 19.1790733, -47.7494049, 47.7482109
14: -41.8267670, 8.8765621, -41.8271103, 8.8769073, -44.1011810, 44.0481186
15: -1.2377033, 28.1101341, -1.2380643, 28.1112976, -24.7158585, 24.6975136
16: -26.6679344, 10.5370674, -26.6666908, 10.5392036, -34.7321854, 34.7060013
17: -39.8772964, 4.8739219, -39.8792114, 4.8740635, -36.0062943, 36.0437927
18: -9.7612658, 26.5557137, -9.7613487, 26.5566292, -36.3178940, 36.3170624
19: -17.4252033, 9.3049335, -17.4256248, 9.3051777, -26.7303810, 26.7305584
20: -25.4211235, 3.7425630, -25.4216557, 3.7432764, -28.8545837, 28.8529129
21: -22.0168610, 8.0695333, -22.0176735, 8.0720453, -30.0889053, 30.0872078
22: -10.4536943, 21.3074608, -10.4539976, 21.3080406, -29.7895889, 29.7985687
23: -10.3461895, 18.1170406, -10.3465366, 18.1181984, -28.2244568, 28.2191849
24: -6.4832954, 22.3731937, -6.4838629, 22.3733578, -28.2662888, 28.2734299
25: -9.5152702, 22.8347855, -9.5160761, 22.8360462, -32.2501678, 32.2764282
26: -23.2086716, 24.3958473, -23.2090607, 24.3969212, -47.0088959, 46.9965057
27: -15.9013634, 17.4263153, -15.9017639, 17.4272938, -33.3286591, 33.3280792
28: -13.9842510, 19.3479996, -13.9847593, 19.3496456, -31.5475693, 31.5458832
29: -10.9108295, 15.4290190, -10.9125214, 15.4289141, -23.1346130, 23.1429214
30: -30.3608246, -0.1045625, -30.3616219, -0.1050549, -27.8738403, 27.8561249
31: -18.1759224, 10.7346497, -18.1767731, 10.7355967, -28.9115181, 28.9114227
32: -51.2237320, -16.1120644, -51.2240906, -16.1095638, -27.8285065, 27.8412323
33: -69.0652771, -12.0236015, -69.0660782, -12.0213299, -49.9347229, 49.9380188
34: -63.1351471, -21.4636993, -63.1354980, -21.4628010, -29.6218567, 29.6300049
35: -42.8272018, -0.5225728, -42.8275375, -0.5238433, -34.4409332, 34.4639587
36: -42.2560463, 2.7683506, -42.2576675, 2.7676857, -36.3522644, 36.3684692
37: -75.2179565, -18.9962807, -75.2195587, -18.9972668, -41.8991241, 41.9181671
38: -52.3657990, 2.0262480, -52.3662872, 2.0260472, -47.3081512, 47.3215485
39: -72.2575302, -13.6649294, -72.2580872, -13.6627369, -54.4878235, 54.4924011
40: -76.4551392, -36.7126694, -76.4565048, -36.7137756, -28.9838791, 28.9887695
41: -52.0065536, -11.1898155, -52.0068893, -11.1886177, -29.1997604, 29.2439423
42: -47.7814331, -16.1895409, -47.7819748, -16.1882515, -24.5224838, 24.5238190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=199, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1730

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0005571, upper bound: 15.0391943
time: 24.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0427704, upper bound: 15.0427704
time: 15.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 42.54 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 42.54
Output dim: 15, lower bound: -14.9858748, upper bound: 15.0390840
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 42.54
Output dim: 15, lower bound: -15.0215078, upper bound: 15.0424561
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 42.54
Output dim: 15, lower bound: -14.9751524, upper bound: 14.9599643
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 42.54
Output dim: 15, lower bound: -15.0236999, upper bound: 14.9663664
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 42.54
Output dim: 15, lower bound: -15.0005571, upper bound: 15.0391943
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 42.54
Output dim: 15, lower bound: -15.0427704, upper bound: 15.0427704

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.6434946, 27.0414314, -5.7515287, 27.0804634, -32.7239571, 32.7929611
1: 0.1138949, 22.8850307, 0.0570240, 22.9028854, -20.5799942, 20.6082153
2: -0.7812879, 25.0639687, -0.8460224, 25.1011162, -23.5494232, 23.5718079
3: -12.5160427, 15.9240341, -12.5784502, 15.9497681, -23.2000122, 23.2743835
4: -6.1554952, 20.8222656, -6.2239413, 20.8306255, -22.9385452, 22.9873276
5: -11.8143673, 20.8991737, -11.8658056, 20.9232101, -30.5505905, 30.5879822
6: -66.1894836, -30.8110542, -66.2131119, -30.7418861, -24.7220230, 24.7282944
7: -17.7924118, 14.9221210, -17.8433228, 14.9350090, -27.3160706, 27.3786011
8: -19.1515503, 10.2320290, -19.2419586, 10.2771397, -24.4892120, 24.5376205
9: -7.1165380, 22.8676720, -7.1599565, 22.8831558, -29.0226288, 29.0369263
10: -31.5304089, 12.2216301, -31.5947609, 12.2628746, -39.0220490, 38.9164429
11: -24.6579170, 0.0996587, -24.7052765, 0.1520958, -23.3515015, 23.3658867
12: -49.0632095, -8.6410217, -49.0757141, -8.5762043, -34.0523453, 34.0074158
13: -28.4698639, 19.0903282, -28.5229340, 19.1635590, -47.6334229, 47.6132622
14: -41.7090912, 8.8289318, -41.7748413, 8.8702507, -43.9717407, 43.8760681
15: -1.0805974, 28.0497265, -1.1665945, 28.1021366, -24.5791321, 24.5767250
16: -26.6154327, 10.5178471, -26.6519852, 10.5306244, -34.6483765, 34.6192017
17: -39.7304916, 4.7979546, -39.8140411, 4.8670931, -35.8251190, 35.8962784
18: -9.6804857, 26.4751987, -9.7521677, 26.5189304, -36.1994171, 36.2273674
19: -17.3679657, 9.1971645, -17.4185028, 9.2540913, -26.6220570, 26.6156673
20: -25.3452187, 3.6174002, -25.4124336, 3.6842833, -28.7213974, 28.7405167
21: -21.9251060, 7.8998399, -22.0083752, 7.9909973, -29.9161034, 29.9082146
22: -10.4161148, 21.2564125, -10.4463387, 21.2850151, -29.6988297, 29.7170258
23: -10.2996435, 18.0398331, -10.3387814, 18.0831947, -28.1445007, 28.1475983
24: -6.4214602, 22.3019199, -6.4733391, 22.3396797, -28.1696472, 28.1874466
25: -9.4587889, 22.7400723, -9.5052757, 22.7922573, -32.1327972, 32.1549072
26: -23.1509171, 24.3265820, -23.1971874, 24.3647785, -46.8591003, 46.8834229
27: -15.8534660, 17.3495445, -15.8908014, 17.3919716, -33.2454376, 33.2403450
28: -13.9355631, 19.2740822, -13.9752789, 19.3161411, -31.4494781, 31.4721222
29: -10.8920746, 15.3729868, -10.9068336, 15.4059448, -23.0830460, 23.0798340
30: -30.2857304, -0.2164524, -30.3543930, -0.1561794, -27.7500000, 27.7768936
31: -18.0974655, 10.5856962, -18.1653786, 10.6648750, -28.7623405, 28.7510757
32: -51.1908951, -16.1872120, -51.2152023, -16.1405640, -27.7579346, 27.7666550
33: -69.0168610, -12.0946922, -69.0516281, -12.0520134, -49.8457489, 49.8552551
34: -63.0860672, -21.5074921, -63.1246796, -21.4806900, -29.5787506, 29.5707397
35: -42.8014374, -0.5655684, -42.8190994, -0.5409012, -34.3695984, 34.4039459
36: -42.2375565, 2.7095299, -42.2507401, 2.7432137, -36.2589722, 36.2987061
37: -75.1846542, -19.0418472, -75.2083740, -19.0152664, -41.8242645, 41.8648224
38: -52.2987404, 1.9207997, -52.3487434, 1.9817200, -47.1782227, 47.1969452
39: -72.2117386, -13.7222881, -72.2436371, -13.6848087, -54.4109802, 54.4178467
40: -76.4046021, -36.7479324, -76.4380875, -36.7237053, -28.9324951, 28.9385681
41: -51.9894066, -11.2494106, -52.0012283, -11.2132788, -29.0922318, 29.1657562
42: -47.7648315, -16.2392883, -47.7773933, -16.2086639, -24.4893494, 24.4999390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9531605, upper bound: 15.0380038
time: 23.68 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9848856, upper bound: 15.0380993
time: 24.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.7242527, 27.0848007, -5.7828751, 27.0824776, -32.8067322, 32.8676758
1: 0.0679777, 22.9231300, 0.0396609, 22.9043713, -20.6265411, 20.6647797
2: -0.8294480, 25.1087837, -0.8658166, 25.1024513, -23.5975876, 23.6363907
3: -12.5691776, 15.9725866, -12.6000586, 15.9510231, -23.2509918, 23.3455276
4: -6.2160110, 20.8502007, -6.2464938, 20.8316479, -22.9985733, 23.0392151
5: -11.8543797, 20.9371185, -11.8810511, 20.9250259, -30.5946808, 30.6426697
6: -66.2377167, -30.7478008, -66.2146225, -30.7187538, -24.7954788, 24.7893600
7: -17.8316269, 14.9490833, -17.8582287, 14.9360676, -27.3582001, 27.4231796
8: -19.2224979, 10.2869740, -19.2707977, 10.2790594, -24.5585175, 24.6228828
9: -7.1603518, 22.8963890, -7.1740394, 22.8848534, -29.0711365, 29.0959396
10: -31.5566273, 12.2522392, -31.5996227, 12.2667332, -39.0578156, 38.9605713
11: -24.7051601, 0.1421874, -24.7079582, 0.1677792, -23.4192581, 23.4081230
12: -49.0819092, -8.5863419, -49.0773773, -8.5590706, -34.0904617, 34.0644379
13: -28.5053902, 19.1369400, -28.5353432, 19.1685181, -47.6739082, 47.6722832
14: -41.7622948, 8.8789272, -41.7924080, 8.8723240, -44.0325623, 43.9503632
15: -1.1475410, 28.1130257, -1.1911926, 28.1053352, -24.6457062, 24.6645432
16: -26.6411762, 10.5444651, -26.6567802, 10.5331459, -34.6876450, 34.6689987
17: -39.7706223, 4.8341026, -39.8275223, 4.8691301, -35.8627014, 35.9448853
18: -9.7449818, 26.5078812, -9.7544432, 26.5310974, -36.2760773, 36.2623253
19: -17.4380817, 9.2453318, -17.4206429, 9.2742224, -26.7123032, 26.6659737
20: -25.4264374, 3.6667442, -25.4156055, 3.7042706, -28.8217850, 28.7898865
21: -22.0124016, 7.9621325, -22.0118313, 8.0166149, -30.0290165, 29.9739647
22: -10.4643536, 21.2836018, -10.4487209, 21.2948780, -29.7618256, 29.7541656
23: -10.3570690, 18.0776215, -10.3412266, 18.0971794, -28.2180939, 28.1864624
24: -6.4838715, 22.3323421, -6.4763794, 22.3514290, -28.2497406, 28.2223434
25: -9.5285778, 22.7837868, -9.5089378, 22.8083744, -32.2225876, 32.2061615
26: -23.2178650, 24.3613605, -23.2004795, 24.3776264, -46.9464569, 46.9331512
27: -15.9016523, 17.3903198, -15.8936176, 17.4073868, -33.3090401, 33.2839355
28: -14.0010929, 19.3161221, -13.9780359, 19.3323936, -31.5307312, 31.5156784
29: -10.9147835, 15.4015198, -10.9079075, 15.4141750, -23.1144257, 23.1075516
30: -30.3560772, -0.1718085, -30.3571091, -0.1401279, -27.8369522, 27.8233185
31: -18.1747398, 10.6461535, -18.1690369, 10.6892262, -28.8639660, 28.8151894
32: -51.2224045, -16.1460991, -51.2175903, -16.1289101, -27.8015823, 27.8080902
33: -69.0709839, -12.0626965, -69.0568314, -12.0412025, -49.9096985, 49.8906403
34: -63.1223907, -21.4908085, -63.1277008, -21.4750900, -29.6271820, 29.5944748
35: -42.8396797, -0.5457137, -42.8218307, -0.5346303, -34.4173279, 34.4297791
36: -42.2820435, 2.7467172, -42.2523651, 2.7565823, -36.3198090, 36.3377151
37: -75.2251816, -19.0214539, -75.2109680, -19.0099411, -41.8713226, 41.8863983
38: -52.4002876, 1.9824324, -52.3538208, 2.0036058, -47.3008118, 47.2597656
39: -72.2646713, -13.7022734, -72.2485046, -13.6805668, -54.4843903, 54.4485168
40: -76.4273376, -36.7335815, -76.4428253, -36.7199936, -28.9577789, 28.9569321
41: -52.0167923, -11.2149582, -52.0027237, -11.2025356, -29.1351700, 29.2000961
42: -47.7856064, -16.2115917, -47.7788200, -16.2007637, -24.5176010, 24.5272141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9889585, upper bound: 15.0414042
time: 21.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0205237, upper bound: 15.0414799
time: 24.16 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.7442093, 27.1179619, -5.6289139, 27.0554008, -32.7996101, 32.7468758
1: 0.0703244, 22.9366989, 0.1347201, 22.8842049, -20.6090012, 20.6051102
2: -0.8461549, 25.1414490, -0.7865970, 25.0809746, -23.5823059, 23.5950546
3: -12.5684452, 15.9954290, -12.4997759, 15.9305496, -23.2271652, 23.2321091
4: -6.2214594, 20.8535862, -6.1491799, 20.8198700, -22.9878006, 22.9538345
5: -11.8516121, 20.9570694, -11.7881508, 20.9040585, -30.5684967, 30.5587234
6: -66.2578583, -30.7525749, -66.1911469, -30.8362942, -24.7501678, 24.7479935
7: -17.8301163, 14.9596233, -17.7731018, 14.9231510, -27.3448257, 27.3282242
8: -19.2240372, 10.3264809, -19.1168365, 10.2441273, -24.5052567, 24.5067787
9: -7.1469860, 22.9078465, -7.0812969, 22.8739052, -29.0364685, 29.0345917
10: -31.5832882, 12.2942791, -31.5383339, 12.2581100, -39.0690536, 39.0389481
11: -24.7390022, 0.1567850, -24.6713734, 0.1060097, -23.3920441, 23.3539124
12: -49.0905151, -8.5813141, -49.0603485, -8.6645107, -34.0081482, 34.0506210
13: -28.5510025, 19.1949005, -28.5233631, 19.1421013, -47.6931038, 47.7182617
14: -41.7580223, 8.9143085, -41.6645126, 8.8409939, -43.9812012, 43.9651489
15: -1.1637259, 28.1545467, -1.0654960, 28.0673637, -24.5895157, 24.6095352
16: -26.6507759, 10.5466890, -26.6060371, 10.5154991, -34.6250763, 34.6408691
17: -39.8260612, 4.8971891, -39.7573166, 4.8413982, -35.9175720, 35.9768829
18: -9.8109941, 26.5186787, -9.7207212, 26.4748745, -36.2858696, 36.2393990
19: -17.4808445, 9.2481852, -17.3861618, 9.1880960, -26.6689415, 26.6343460
20: -25.4825325, 3.6888447, -25.3727722, 3.6298292, -28.8138809, 28.7506409
21: -22.0806713, 8.0006542, -21.9619160, 7.9274712, -30.0081425, 29.9625702
22: -10.4847527, 21.2805119, -10.4248686, 21.2494984, -29.7419739, 29.6984863
23: -10.3873959, 18.0897694, -10.3155375, 18.0543766, -28.2005157, 28.1533813
24: -6.5277920, 22.3387260, -6.4442635, 22.2985382, -28.2395782, 28.1818085
25: -9.5615320, 22.7975025, -9.4742298, 22.7517738, -32.2049789, 32.1465302
26: -23.2547569, 24.3551197, -23.1700058, 24.3106155, -46.9407501, 46.8689423
27: -15.9325457, 17.3957634, -15.8804197, 17.3593063, -33.2918510, 33.2761841
28: -14.0318699, 19.3101006, -13.9494877, 19.2674599, -31.5241547, 31.4776077
29: -10.9226627, 15.4150915, -10.9012928, 15.3907070, -23.1074219, 23.1071701
30: -30.4131622, -0.1354327, -30.3215485, -0.1796830, -27.8543091, 27.7826843
31: -18.2289181, 10.6680746, -18.1220303, 10.5946140, -28.8235321, 28.7901039
32: -51.2421379, -16.1469345, -51.1952858, -16.2117519, -27.7678833, 27.7744827
33: -69.0880737, -12.0890713, -69.0014496, -12.1654310, -49.8199005, 49.8050842
34: -63.1530838, -21.5015278, -63.0943489, -21.5475540, -29.5691910, 29.5458832
35: -42.8480835, -0.5725224, -42.7839127, -0.6328268, -34.3674011, 34.3662491
36: -42.2885017, 2.7064886, -42.2210999, 2.6305959, -36.2636108, 36.2647247
37: -75.2419357, -19.0427074, -75.1840057, -19.0931892, -41.8409424, 41.8301773
38: -52.4374771, 1.9397526, -52.2894363, 1.8249245, -47.2038269, 47.1553345
39: -72.2812500, -13.7398396, -72.1915054, -13.8216553, -54.3787537, 54.3541107
40: -76.4512405, -36.7260551, -76.4252014, -36.7535248, -28.9404068, 28.9439850
41: -52.0252838, -11.2323856, -51.9840240, -11.2921124, -29.1528015, 29.1717377
42: -47.7934723, -16.2063274, -47.7671394, -16.2398109, -24.4847260, 24.4910851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=144, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9910434, upper bound: 14.9652533
time: 36.37 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0227030, upper bound: 14.9653725
time: 18.46 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.7574425, 27.0809441, -5.8051643, 27.0843887, -32.8418312, 32.8861084
1: 0.0629978, 22.9035759, 0.0343993, 22.9060631, -20.6381683, 20.6649704
2: -0.8489645, 25.1015472, -0.8788824, 25.1040840, -23.6088257, 23.6438980
3: -12.5739403, 15.9510136, -12.6067009, 15.9531898, -23.2602615, 23.2825546
4: -6.2170181, 20.8306980, -6.2527208, 20.8325748, -22.9983749, 23.0288391
5: -11.8647156, 20.9244595, -11.8898735, 20.9275837, -30.6081238, 30.6333313
6: -66.2137146, -30.7504044, -66.2166672, -30.7133293, -24.7856140, 24.7834015
7: -17.8382950, 14.9359598, -17.8628159, 14.9377747, -27.3719788, 27.4011612
8: -19.2434635, 10.2792959, -19.2878304, 10.2823334, -24.5657654, 24.5951920
9: -7.1512861, 22.8866367, -7.1730866, 22.8913536, -29.0619202, 29.0931931
10: -31.5937901, 12.2792988, -31.6065102, 12.2887859, -39.1198425, 39.0932007
11: -24.7041912, 0.1537900, -24.7090492, 0.1778495, -23.4144974, 23.3903465
12: -49.0775375, -8.5794106, -49.0800972, -8.5511007, -34.0930023, 34.0740967
13: -28.5348148, 19.1642437, -28.5543442, 19.1729965, -47.7078094, 47.7185898
14: -41.7758751, 8.8706112, -41.8047066, 8.8742886, -44.0403290, 44.0124969
15: -1.1718903, 28.1011086, -1.2090411, 28.1073112, -24.6444931, 24.6602249
16: -26.6494102, 10.5299129, -26.6585693, 10.5361052, -34.6945648, 34.6826019
17: -39.8393173, 4.8682766, -39.8624840, 4.8715487, -35.9666901, 36.0216675
18: -9.7550335, 26.5215950, -9.7586060, 26.5416451, -36.2966766, 36.2802010
19: -17.4188900, 9.2517433, -17.4228706, 9.2815886, -26.7004776, 26.6746140
20: -25.4120979, 3.6891880, -25.4177094, 3.7197688, -28.8241119, 28.7976913
21: -22.0066605, 8.0015554, -22.0131817, 8.0420761, -30.0487366, 30.0147362
22: -10.4466782, 21.2782764, -10.4508667, 21.2951145, -29.7663193, 29.7570419
23: -10.3391304, 18.0799255, -10.3434753, 18.1017075, -28.2011566, 28.1788635
24: -6.4745626, 22.3408947, -6.4800200, 22.3590946, -28.2429810, 28.2349548
25: -9.5046949, 22.7901115, -9.5114574, 22.8163528, -32.2189331, 32.2205963
26: -23.1989708, 24.3576069, -23.2048645, 24.3800354, -46.9761505, 46.9404755
27: -15.8934097, 17.3851700, -15.8982763, 17.4091225, -33.3025322, 33.2834473
28: -13.9763479, 19.3042393, -13.9812698, 19.3303585, -31.5209503, 31.4992294
29: -10.9072628, 15.4029484, -10.9109936, 15.4173679, -23.1192398, 23.1145096
30: -30.3529243, -0.1472745, -30.3581314, -0.1239412, -27.8486633, 27.8110428
31: -18.1654320, 10.6699953, -18.1722031, 10.7070913, -28.8725243, 28.8421974
32: -51.2171059, -16.1448631, -51.2211876, -16.1240292, -27.8080978, 27.8050461
33: -69.0507050, -12.0589008, -69.0597305, -12.0369978, -49.9036407, 49.8957062
34: -63.1267891, -21.4811230, -63.1317749, -21.4705448, -29.6040497, 29.6020966
35: -42.8194504, -0.5449755, -42.8241234, -0.5337012, -34.4209137, 34.4348907
36: -42.2510147, 2.7285986, -42.2554855, 2.7500703, -36.3279419, 36.3232193
37: -75.2106171, -19.0214233, -75.2163925, -19.0088291, -41.8792114, 41.8893356
38: -52.3520889, 1.9662261, -52.3601341, 1.9995332, -47.2687073, 47.2548981
39: -72.2440186, -13.6923485, -72.2522583, -13.6749020, -54.4604034, 54.4555054
40: -76.4416351, -36.7232704, -76.4505005, -36.7184792, -28.9663315, 28.9723282
41: -52.0021057, -11.2227383, -52.0049286, -11.2031279, -29.1789246, 29.2079697
42: -47.7772446, -16.2131786, -47.7801285, -16.1987305, -24.5086899, 24.4985352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9678626, upper bound: 15.0381021
time: 20.93 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9995670, upper bound: 15.0382042
time: 17.53 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.8384876, 27.1241779, -5.8365574, 27.0863800, -32.9248657, 32.9607353
1: 0.0169582, 22.9416466, 0.0169611, 22.9075508, -20.6848297, 20.7215271
2: -0.8973169, 25.1463604, -0.8987253, 25.1053658, -23.6574326, 23.7084732
3: -12.6273642, 15.9995527, -12.6283417, 15.9544706, -23.3114929, 23.3537598
4: -6.2783704, 20.8586502, -6.2756009, 20.8335915, -23.0587082, 23.0809174
5: -11.9049540, 20.9626236, -11.9051781, 20.9294262, -30.6525650, 30.6880188
6: -66.2616730, -30.6869411, -66.2182617, -30.6900272, -24.8592300, 24.8448601
7: -17.8776932, 14.9629288, -17.8777466, 14.9388313, -27.4144669, 27.4457779
8: -19.3157120, 10.3346748, -19.3172874, 10.2842855, -24.6353760, 24.6805534
9: -7.1944022, 22.9126873, -7.1872134, 22.8930321, -29.1106262, 29.1523209
10: -31.6163368, 12.3076324, -31.6114292, 12.2926426, -39.1512909, 39.1356583
11: -24.7514572, 0.1964078, -24.7117558, 0.1936185, -23.4824448, 23.4327888
12: -49.0965157, -8.5237236, -49.0817528, -8.5337772, -34.1328583, 34.1322861
13: -28.5709801, 19.2101841, -28.5668488, 19.1778793, -47.7488594, 47.7770309
14: -41.8296204, 8.9203453, -41.8223572, 8.8763695, -44.1022797, 44.0876923
15: -1.2390652, 28.1642189, -1.2337666, 28.1105118, -24.7115326, 24.7479286
16: -26.6756096, 10.5561323, -26.6633682, 10.5385609, -34.7337646, 34.7329636
17: -39.8800735, 4.9047575, -39.8761101, 4.8735380, -36.0060425, 36.0727768
18: -9.8194876, 26.5543633, -9.7608232, 26.5538177, -36.3733063, 36.3151855
19: -17.4888649, 9.3000746, -17.4249878, 9.3017797, -26.7906456, 26.7250633
20: -25.4933319, 3.7386551, -25.4208183, 3.7398419, -28.9245758, 28.8473892
21: -22.0939560, 8.0639629, -22.0166321, 8.0677462, -30.1617012, 30.0805950
22: -10.4948235, 21.3056030, -10.4533024, 21.3051128, -29.8293610, 29.7943039
23: -10.3964863, 18.1179047, -10.3459520, 18.1158142, -28.2746964, 28.2179718
24: -6.5369864, 22.3714333, -6.4830685, 22.3709297, -28.3231812, 28.2698975
25: -9.5745707, 22.8339729, -9.5151014, 22.8325844, -32.3088226, 32.2719650
26: -23.2659893, 24.3927593, -23.2082062, 24.3930798, -47.0635529, 46.9907684
27: -15.9413633, 17.4261513, -15.9010868, 17.4245758, -33.3659401, 33.3272400
28: -14.0417547, 19.3465538, -13.9840336, 19.3467026, -31.6020966, 31.5431824
29: -10.9282532, 15.4306765, -10.9120865, 15.4257565, -23.1498795, 23.1425629
30: -30.4233208, -0.1024623, -30.3608322, -0.1078849, -27.9357834, 27.8576431
31: -18.2426319, 10.7305813, -18.1758060, 10.7315044, -28.9741364, 28.9063873
32: -51.2490234, -16.1035385, -51.2235260, -16.1123772, -27.8523026, 27.8467178
33: -69.1044693, -12.0262957, -69.0649185, -12.0262337, -49.9671021, 49.9319763
34: -63.1630592, -21.4641972, -63.1348267, -21.4648895, -29.6524506, 29.6259842
35: -42.8577538, -0.5247018, -42.8268547, -0.5274291, -34.4683990, 34.4612122
36: -42.2948685, 2.7662930, -42.2571144, 2.7636080, -36.3885345, 36.3626251
37: -75.2501526, -19.0020180, -75.2189941, -19.0034065, -41.9266052, 41.9103317
38: -52.4535446, 2.0280190, -52.3652496, 2.0215206, -47.3914185, 47.3180237
39: -72.2968063, -13.6719332, -72.2569656, -13.6706543, -54.5352478, 54.4874115
40: -76.4646835, -36.7086906, -76.4552460, -36.7147331, -28.9918365, 28.9908829
41: -52.0293732, -11.1881294, -52.0064507, -11.1923037, -29.2200775, 29.2423553
42: -47.7979927, -16.1852150, -47.7815247, -16.1907635, -24.5371780, 24.5264587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=199, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0102863, upper bound: 15.0417214
time: 21.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0417971, upper bound: 15.0417972
time: 32.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 56.79 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -14.9531605, upper bound: 15.0380038
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -14.9848856, upper bound: 15.0380993
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -14.9889585, upper bound: 15.0414042
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -15.0205237, upper bound: 15.0414799
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 56.79
Output dim: 15, lower bound: -14.9910434, upper bound: 14.9652533
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -15.0227030, upper bound: 14.9653725
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -14.9678626, upper bound: 15.0381021
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -14.9995670, upper bound: 15.0382042
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -15.0102863, upper bound: 15.0417214
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.79
Output dim: 15, lower bound: -15.0417971, upper bound: 15.0417972

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.6389103, 27.0317249, -5.7386503, 27.0536728, -32.6925812, 32.7703743
1: 0.1170387, 22.8736725, 0.0659564, 22.8703804, -20.5400009, 20.5865784
2: -0.7802043, 25.0487518, -0.8430519, 25.0576000, -23.5037613, 23.5528641
3: -12.5148554, 15.9161797, -12.5750275, 15.9274197, -23.1765518, 23.2627029
4: -6.1532121, 20.8079205, -6.2175612, 20.7896042, -22.8949661, 22.9669342
5: -11.8131008, 20.8872337, -11.8620138, 20.8900757, -30.5163193, 30.5719376
6: -66.1805038, -30.8139648, -66.1877213, -30.7502956, -24.7055206, 24.6983185
7: -17.7899990, 14.9108524, -17.8364868, 14.9026375, -27.2801132, 27.3603058
8: -19.1507950, 10.2144833, -19.2398338, 10.2270861, -24.4376755, 24.5167007
9: -7.1132159, 22.8577900, -7.1506438, 22.8557510, -28.9905624, 29.0195618
10: -31.5266304, 12.2147121, -31.5840092, 12.2431126, -38.9972534, 38.8980331
11: -24.6375446, 0.0978901, -24.6495190, 0.1471758, -23.3274078, 23.3065910
12: -49.0515099, -8.6444416, -49.0427933, -8.5859261, -34.0295868, 33.9698792
13: -28.4660339, 19.0733509, -28.5122108, 19.1157570, -47.5817909, 47.5855637
14: -41.7057953, 8.8240318, -41.7656250, 8.8562222, -43.9532928, 43.8608551
15: -1.0769477, 28.0355949, -1.1562839, 28.0616837, -24.5332642, 24.5519829
16: -26.6087475, 10.5073395, -26.6330910, 10.5014114, -34.6092834, 34.5893326
17: -39.7185516, 4.7966423, -39.7798615, 4.8634100, -35.8057938, 35.8515549
18: -9.6546240, 26.4740295, -9.6781025, 26.5156097, -36.1702347, 36.1521301
19: -17.3547592, 9.1970415, -17.3814487, 9.2537003, -26.6084595, 26.5784912
20: -25.3314934, 3.6153266, -25.3736115, 3.6785145, -28.7002716, 28.6951370
21: -21.9078407, 7.8988914, -21.9597759, 7.9883628, -29.8962040, 29.8586674
22: -10.4027767, 21.2557945, -10.4081717, 21.2832165, -29.6808624, 29.6719284
23: -10.2819347, 18.0379257, -10.2882032, 18.0778179, -28.1186752, 28.0893478
24: -6.4067445, 22.3008156, -6.4319849, 22.3365765, -28.1502686, 28.1418152
25: -9.4444199, 22.7383232, -9.4647331, 22.7872715, -32.1093903, 32.1030273
26: -23.1259480, 24.3259659, -23.1259022, 24.3630562, -46.8313599, 46.8099823
27: -15.8452024, 17.3474140, -15.8681707, 17.3859043, -33.2311058, 33.2155838
28: -13.9159307, 19.2731247, -13.9191704, 19.3133526, -31.4266281, 31.4145355
29: -10.8801384, 15.3723860, -10.8728085, 15.4041080, -23.0685959, 23.0444336
30: -30.2642593, -0.2195327, -30.2948513, -0.1648498, -27.7178726, 27.7088470
31: -18.0853310, 10.5848579, -18.1312199, 10.6625376, -28.7478676, 28.7160778
32: -51.1807022, -16.1925468, -51.1873322, -16.1558762, -27.7332306, 27.7322617
33: -69.0129089, -12.0977383, -69.0403671, -12.0608025, -49.8318787, 49.8401337
34: -63.0671425, -21.5106030, -63.0712509, -21.4895363, -29.5507431, 29.5109940
35: -42.7900772, -0.5675211, -42.7869186, -0.5464926, -34.3521729, 34.3700562
36: -42.2273331, 2.7076969, -42.2218399, 2.7379949, -36.2436066, 36.2681580
37: -75.1716461, -19.0433350, -75.1716385, -19.0195751, -41.8052216, 41.8258209
38: -52.2770309, 1.9161382, -52.2866592, 1.9687281, -47.1444855, 47.1306610
39: -72.2078705, -13.7289982, -72.2328186, -13.7036915, -54.3874664, 54.3993530
40: -76.3985367, -36.7523460, -76.4211502, -36.7361984, -28.9094162, 28.9135895
41: -51.9809875, -11.2517958, -51.9776611, -11.2201881, -29.0765457, 29.1370773
42: -47.7555008, -16.2421951, -47.7520180, -16.2169342, -24.4714813, 24.4666138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9446164, upper bound: 15.0052804
time: 22.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9446164, upper bound: 15.0297336
time: 27.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.6422701, 27.0405846, -5.8630333, 27.0873299, -32.7295990, 32.9036179
1: 0.1150980, 22.8838081, -0.0056982, 22.9101677, -20.5767441, 20.6764984
2: -0.7808731, 25.0610218, -0.9273231, 25.1052933, -23.5525360, 23.6537018
3: -12.5156403, 15.9224453, -12.6293926, 15.9544907, -23.2062378, 23.3266068
4: -6.1548471, 20.8192921, -6.3183746, 20.8329468, -22.9422760, 23.0826721
5: -11.8140278, 20.8972111, -11.9311676, 20.9298191, -30.5568924, 30.6546478
6: -66.1851120, -30.8117676, -66.2151031, -30.6727943, -24.7996216, 24.7219124
7: -17.7913380, 14.9199257, -17.8998814, 14.9389143, -27.3190842, 27.4369507
8: -19.1513557, 10.2282848, -19.3342609, 10.2829323, -24.4935303, 24.6270943
9: -7.1156640, 22.8652000, -7.2438812, 22.8831825, -29.0195312, 29.1215973
10: -31.5293674, 12.2157459, -31.6338005, 12.2665606, -39.0303040, 38.9627686
11: -24.6552372, 0.0991521, -24.7186279, 0.2305243, -23.4262390, 23.3722725
12: -49.0605316, -8.6421089, -49.0804596, -8.4958611, -34.1293182, 34.0088959
13: -28.4687462, 19.0861092, -28.5888996, 19.1772270, -47.6459732, 47.6750107
14: -41.7081261, 8.8260031, -41.8245468, 8.8768120, -43.9812927, 43.9261475
15: -1.0799475, 28.0476398, -1.2546597, 28.1133595, -24.5800629, 24.6651154
16: -26.6130486, 10.5139189, -26.6998062, 10.5326757, -34.6464996, 34.6735382
17: -39.7231178, 4.7975540, -39.8309479, 4.8981652, -35.8605194, 35.9133453
18: -9.6765337, 26.4749203, -9.7583895, 26.5886917, -36.2652245, 36.2333107
19: -17.3656273, 9.1971684, -17.4287701, 9.2996988, -26.6653252, 26.6259384
20: -25.3439522, 3.6169279, -25.4275436, 3.7492802, -28.7889252, 28.7487259
21: -21.9222031, 7.8997025, -22.0231857, 8.0640059, -29.9862099, 29.9228878
22: -10.4130545, 21.2561817, -10.4549007, 21.3269539, -29.7612305, 29.7121811
23: -10.2967119, 18.0394020, -10.3494453, 18.1473732, -28.2156677, 28.1462784
24: -6.4197669, 22.3016033, -6.4863629, 22.3789959, -28.2122269, 28.1922836
25: -9.4555273, 22.7397537, -9.5180464, 22.8344917, -32.2002258, 32.1506958
26: -23.1471996, 24.3263512, -23.2084427, 24.4514694, -46.9483185, 46.8912354
27: -15.8488913, 17.3490982, -15.8939705, 17.4297905, -33.2786827, 33.2430687
28: -13.9324713, 19.2739182, -13.9873276, 19.3912106, -31.5226135, 31.4811630
29: -10.8892784, 15.3727684, -10.9129639, 15.4567480, -23.1357574, 23.0832901
30: -30.2832146, -0.2169328, -30.3676891, -0.0916231, -27.8130646, 27.7818832
31: -18.0929241, 10.5855112, -18.1787930, 10.7076273, -28.8005524, 28.7643051
32: -51.1894989, -16.1886730, -51.2270737, -16.0801926, -27.8206940, 27.7695465
33: -69.0153351, -12.0956383, -69.0666962, -12.0282135, -49.8688354, 49.8711090
34: -63.0835648, -21.5088692, -63.1363144, -21.4017162, -29.6590958, 29.5696793
35: -42.7988091, -0.5664368, -42.8262558, -0.5029566, -34.4024963, 34.4100418
36: -42.2355576, 2.7088153, -42.2551994, 2.8001838, -36.3150635, 36.3020325
37: -75.1820526, -19.0423508, -75.2155380, -18.9841938, -41.8527222, 41.8737030
38: -52.2944679, 1.9187593, -52.3601875, 2.0628452, -47.2573395, 47.2032623
39: -72.2105484, -13.7288189, -72.2587280, -13.6864958, -54.4163971, 54.4289093
40: -76.4024658, -36.7502975, -76.4478989, -36.7128029, -28.9368668, 28.9429626
41: -51.9872284, -11.2501469, -52.0060883, -11.1519642, -29.1527405, 29.1652451
42: -47.7631187, -16.2401352, -47.7829018, -16.1588745, -24.5501480, 24.4970703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9764498, upper bound: 15.0053759
time: 15.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9764498, upper bound: 15.0298402
time: 22.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.7197208, 27.0750847, -5.7699871, 27.0556889, -32.7754097, 32.8450699
1: 0.0710826, 22.9117546, 0.0486219, 22.8717995, -20.5865860, 20.6431427
2: -0.8283646, 25.0935383, -0.8628349, 25.0589237, -23.5519562, 23.6174545
3: -12.5680218, 15.9647675, -12.5966082, 15.9286938, -23.2275467, 23.3338547
4: -6.2137547, 20.8358459, -6.2401319, 20.7906322, -22.9549866, 23.0188141
5: -11.8530722, 20.9252052, -11.8772793, 20.8919430, -30.5603561, 30.6265793
6: -66.2287140, -30.7507496, -66.1892776, -30.7271309, -24.7790146, 24.7594452
7: -17.8292160, 14.9377861, -17.8513794, 14.9036961, -27.3222351, 27.4048843
8: -19.2218018, 10.2694988, -19.2687035, 10.2289810, -24.5069962, 24.6019516
9: -7.1570501, 22.8864708, -7.1647496, 22.8574677, -29.0390930, 29.0785980
10: -31.5528793, 12.2453432, -31.5889015, 12.2469893, -39.0330505, 38.9422607
11: -24.6847935, 0.1403632, -24.6521626, 0.1628633, -23.3951721, 23.3488846
12: -49.0701599, -8.5897818, -49.0444260, -8.5688744, -34.0677338, 34.0268784
13: -28.5016441, 19.1199303, -28.5246544, 19.1206856, -47.6223297, 47.6445847
14: -41.7591095, 8.8739920, -41.7831650, 8.8583088, -44.0141296, 43.9351425
15: -1.1439610, 28.0989189, -1.1808949, 28.0649052, -24.5998611, 24.6398163
16: -26.6346016, 10.5340118, -26.6379814, 10.5039186, -34.6486359, 34.6391678
17: -39.7586784, 4.8328047, -39.7933502, 4.8654175, -35.8433685, 35.9001236
18: -9.7191534, 26.5067406, -9.6803522, 26.5277672, -36.2469215, 36.1870918
19: -17.4248581, 9.2451801, -17.3836136, 9.2738314, -26.6986885, 26.6287937
20: -25.4127026, 3.6647286, -25.3767471, 3.6985240, -28.8005981, 28.7445068
21: -21.9951115, 7.9612074, -21.9631920, 8.0139847, -30.0090961, 29.9244003
22: -10.4509735, 21.2830200, -10.4105721, 21.2931595, -29.7438965, 29.7090607
23: -10.3394489, 18.0757179, -10.2906590, 18.0918579, -28.1923523, 28.1282349
24: -6.4691758, 22.3312378, -6.4349761, 22.3483276, -28.2303314, 28.1767197
25: -9.5142498, 22.7820244, -9.4683857, 22.8033829, -32.1991882, 32.1542435
26: -23.1928692, 24.3606625, -23.1294308, 24.3759823, -46.9188080, 46.8596039
27: -15.8934145, 17.3881607, -15.8709869, 17.4012356, -33.2946510, 33.2591476
28: -13.9814425, 19.3151703, -13.9219189, 19.3296051, -31.5079041, 31.4580688
29: -10.9028254, 15.4008789, -10.8738804, 15.4123878, -23.0999985, 23.0721130
30: -30.3345928, -0.1748636, -30.2975464, -0.1488185, -27.8048172, 27.7552567
31: -18.1626320, 10.6453285, -18.1348495, 10.6868792, -28.8495102, 28.7801781
32: -51.2121964, -16.1514378, -51.1897011, -16.1441936, -27.7768173, 27.7736969
33: -69.0669785, -12.0657587, -69.0455322, -12.0499907, -49.8957214, 49.8755646
34: -63.1034851, -21.4939060, -63.0743256, -21.4838905, -29.5991974, 29.5347595
35: -42.8283272, -0.5476828, -42.7896919, -0.5402546, -34.3998871, 34.3958740
36: -42.2718391, 2.7449462, -42.2236404, 2.7514558, -36.3045502, 36.3071747
37: -75.2121124, -19.0229778, -75.1743011, -19.0142174, -41.8522644, 41.8473129
38: -52.3786163, 1.9778509, -52.2915993, 1.9905663, -47.2670593, 47.1934967
39: -72.2608185, -13.7087803, -72.2376709, -13.6993866, -54.4610748, 54.4300079
40: -76.4213028, -36.7379303, -76.4258728, -36.7325058, -28.9346542, 28.9319839
41: -52.0083618, -11.2173700, -51.9791870, -11.2094288, -29.1194763, 29.1714478
42: -47.7762146, -16.2144833, -47.7534256, -16.2089939, -24.4997101, 24.4938774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9805841, upper bound: 15.0086862
time: 20.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9805841, upper bound: 15.0331939
time: 30.49 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.7230597, 27.0839024, -5.8943777, 27.0892639, -32.8123245, 32.9782791
1: 0.0691462, 22.9218903, -0.0230949, 22.9116650, -20.6233673, 20.7330475
2: -0.8290517, 25.1058216, -0.9471025, 25.1066303, -23.6006622, 23.7182770
3: -12.5688295, 15.9710226, -12.6509800, 15.9557581, -23.2572479, 23.3977966
4: -6.2153935, 20.8472347, -6.3409185, 20.8339462, -23.0022736, 23.1345787
5: -11.8539906, 20.9351959, -11.9463959, 20.9316559, -30.6008377, 30.7092667
6: -66.2333374, -30.7485924, -66.2165985, -30.6495934, -24.8731155, 24.7829895
7: -17.8305988, 14.9468231, -17.9147644, 14.9399567, -27.3611755, 27.4814911
8: -19.2222919, 10.2832737, -19.3631210, 10.2848558, -24.5627899, 24.7123337
9: -7.1594687, 22.8938866, -7.2579927, 22.8848839, -29.0680466, 29.1806488
10: -31.5556335, 12.2464247, -31.6386814, 12.2704678, -39.0661316, 39.0070648
11: -24.7024956, 0.1416371, -24.7212715, 0.2461934, -23.4940186, 23.4145432
12: -49.0792236, -8.5874519, -49.0821228, -8.4787397, -34.1674194, 34.0658493
13: -28.5043716, 19.1326904, -28.6013298, 19.1821060, -47.6864777, 47.7340202
14: -41.7613525, 8.8759842, -41.8420563, 8.8788776, -44.0421143, 44.0005341
15: -1.1469040, 28.1109772, -1.2792730, 28.1165390, -24.6466217, 24.7529297
16: -26.6388092, 10.5405226, -26.7045841, 10.5352154, -34.6857910, 34.7233429
17: -39.7633476, 4.8337345, -39.8444366, 4.9001656, -35.8980560, 35.9619141
18: -9.7410450, 26.5075569, -9.7606506, 26.6008625, -36.3419075, 36.2682076
19: -17.4357300, 9.2453413, -17.4309101, 9.3197908, -26.7555199, 26.6762505
20: -25.4251842, 3.6662784, -25.4306793, 3.7692847, -28.8893356, 28.7980881
21: -22.0095215, 7.9619665, -22.0265713, 8.0896606, -30.0991821, 29.9885368
22: -10.4612255, 21.2834339, -10.4572849, 21.3369236, -29.8241806, 29.7492828
23: -10.3542242, 18.0772209, -10.3519211, 18.1614132, -28.2893829, 28.1851959
24: -6.4821119, 22.3320732, -6.4893398, 22.3907280, -28.2923737, 28.2271576
25: -9.5253677, 22.7834110, -9.5217257, 22.8506432, -32.2900238, 32.2018585
26: -23.2141590, 24.3611565, -23.2118168, 24.4643364, -47.0357971, 46.9408722
27: -15.8971519, 17.3898315, -15.8968325, 17.4451942, -33.3423462, 33.2866631
28: -13.9980249, 19.3160172, -13.9900169, 19.4074650, -31.6038361, 31.5247574
29: -10.9119673, 15.4012814, -10.9140244, 15.4650478, -23.1671677, 23.1109772
30: -30.3536034, -0.1722839, -30.3703842, -0.0755670, -27.9000320, 27.8282700
31: -18.1701660, 10.6459560, -18.1824226, 10.7319660, -28.9021320, 28.8283787
32: -51.2210197, -16.1475506, -51.2294159, -16.0684910, -27.8643570, 27.8109818
33: -69.0694427, -12.0635357, -69.0719299, -12.0174465, -49.9326935, 49.9065704
34: -63.1198692, -21.4921055, -63.1392746, -21.3961182, -29.7075577, 29.5934143
35: -42.8370667, -0.5466075, -42.8289719, -0.4967444, -34.4502411, 34.4358063
36: -42.2800598, 2.7461131, -42.2568932, 2.8135731, -36.3759460, 36.3410492
37: -75.2226181, -19.0220356, -75.2181244, -18.9788170, -41.8997955, 41.8952179
38: -52.3961105, 1.9804649, -52.3651733, 2.0847111, -47.3799438, 47.2659912
39: -72.2635880, -13.7086258, -72.2635651, -13.6822538, -54.4898834, 54.4595337
40: -76.4251556, -36.7359238, -76.4526138, -36.7091026, -28.9621048, 28.9613342
41: -52.0145874, -11.2157402, -52.0076447, -11.1411762, -29.1957550, 29.1995697
42: -47.7838974, -16.2123985, -47.7843552, -16.1508942, -24.5783539, 24.5243301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0122627, upper bound: 15.0087678
time: 26.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0122627, upper bound: 15.0332837
time: 25.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.7430134, 27.1170807, -5.7403851, 27.0622215, -32.8052368, 32.8574677
1: 0.0715156, 22.9354725, 0.0720031, 22.8915272, -20.6057739, 20.6733932
2: -0.8457036, 25.1385117, -0.8678393, 25.0852013, -23.5854187, 23.6769257
3: -12.5680695, 15.9938335, -12.5506973, 15.9352512, -23.2334366, 23.2843170
4: -6.2207918, 20.8506565, -6.2435327, 20.8222008, -22.9915390, 23.0490723
5: -11.8512363, 20.9551811, -11.8534021, 20.9106369, -30.5747147, 30.6253052
6: -66.2534485, -30.7533932, -66.1931381, -30.7672634, -24.8277283, 24.7416077
7: -17.8290825, 14.9573631, -17.8296871, 14.9270763, -27.3478241, 27.3865814
8: -19.2237587, 10.3227615, -19.2091141, 10.2499218, -24.5096588, 24.5962181
9: -7.1460786, 22.9053936, -7.1651459, 22.8740044, -29.0333557, 29.1192169
10: -31.5823021, 12.2883854, -31.5771999, 12.2618675, -39.0773392, 39.0850906
11: -24.7363815, 0.1562393, -24.6848011, 0.1843953, -23.4667435, 23.3603859
12: -49.0878220, -8.5823879, -49.0650787, -8.5842409, -34.0849457, 34.0521317
13: -28.5499916, 19.1906548, -28.5892029, 19.1557102, -47.7057037, 47.7798576
14: -41.7571411, 8.9113083, -41.7140694, 8.8475475, -43.9907227, 44.0151520
15: -1.1631045, 28.1525517, -1.1534681, 28.0785866, -24.5904465, 24.6979141
16: -26.6483784, 10.5427694, -26.6538639, 10.5176277, -34.6232300, 34.6951599
17: -39.8187599, 4.8969297, -39.7742195, 4.8724566, -35.9528580, 35.9939804
18: -9.8070831, 26.5183716, -9.7269392, 26.5446320, -36.3517151, 36.2453117
19: -17.4785004, 9.2481747, -17.3964386, 9.2336607, -26.7121620, 26.6446133
20: -25.4812698, 3.6883764, -25.3879375, 3.6947870, -28.8814087, 28.7589035
21: -22.0777912, 8.0004597, -21.9767227, 8.0005035, -30.0782948, 29.9771824
22: -10.4817076, 21.2803307, -10.4333725, 21.2914352, -29.8043594, 29.6936340
23: -10.3844671, 18.0894051, -10.3262539, 18.1185341, -28.2716599, 28.1521225
24: -6.5260868, 22.3384361, -6.4572825, 22.3378086, -28.2821350, 28.1866608
25: -9.5582685, 22.7971287, -9.4869709, 22.7940559, -32.2724228, 32.1422958
26: -23.2510662, 24.3548584, -23.1812935, 24.3972778, -47.0300293, 46.8767242
27: -15.9280081, 17.3952770, -15.8835897, 17.3971252, -33.3251343, 33.2788658
28: -14.0287819, 19.3099213, -13.9615278, 19.3424892, -31.5972290, 31.4867783
29: -10.9198446, 15.4148550, -10.9074116, 15.4414635, -23.1601639, 23.1105576
30: -30.4106674, -0.1359363, -30.3349590, -0.1152828, -27.9173508, 27.7877197
31: -18.2243786, 10.6678190, -18.1354561, 10.6372805, -28.8616600, 28.8032761
32: -51.2407265, -16.1484470, -51.2071762, -16.1514301, -27.8306274, 27.7773819
33: -69.0864944, -12.0900068, -69.0165863, -12.1417131, -49.8428802, 49.8209686
34: -63.1504669, -21.5028687, -63.1059875, -21.4685612, -29.6496048, 29.5448456
35: -42.8454971, -0.5733504, -42.7910690, -0.5950093, -34.4002533, 34.3723526
36: -42.2864914, 2.7057869, -42.2255859, 2.6875653, -36.3196869, 36.2680130
37: -75.2394257, -19.0433731, -75.1912155, -19.0620193, -41.8693848, 41.8390732
38: -52.4331894, 1.9377747, -52.3008575, 1.9060674, -47.2829742, 47.1616821
39: -72.2800903, -13.7463121, -72.2065811, -13.8234348, -54.3840637, 54.3650970
40: -76.4490356, -36.7283936, -76.4350510, -36.7426262, -28.9447784, 28.9484253
41: -52.0230293, -11.2330856, -51.9889679, -11.2309132, -29.2131805, 29.1712646
42: -47.7917786, -16.2071419, -47.7726974, -16.1901531, -24.5454483, 24.4882622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=144, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0142094, upper bound: 14.9326440
time: 8.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0142094, upper bound: 14.9566196
time: 9.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.7529449, 27.0712051, -5.7922993, 27.0576324, -32.8105774, 32.8635025
1: 0.0661476, 22.8922348, 0.0433400, 22.8735142, -20.5981598, 20.6433258
2: -0.8478994, 25.0863323, -0.8758924, 25.0605202, -23.5631943, 23.6249542
3: -12.5726843, 15.9431581, -12.6032619, 15.9308548, -23.2368546, 23.2708588
4: -6.2147875, 20.8163757, -6.2463121, 20.7915573, -22.9547806, 23.0084763
5: -11.8634014, 20.9126053, -11.8860855, 20.8944340, -30.5738525, 30.6173172
6: -66.2047195, -30.7533264, -66.1913147, -30.7217083, -24.7691345, 24.7534485
7: -17.8358860, 14.9246540, -17.8559742, 14.9054384, -27.3360291, 27.3828506
8: -19.2426720, 10.2617645, -19.2856865, 10.2323017, -24.5142517, 24.5743065
9: -7.1479454, 22.8767757, -7.1637554, 22.8639355, -29.0298080, 29.0758362
10: -31.5900192, 12.2723513, -31.5958176, 12.2689705, -39.0950394, 39.0747833
11: -24.6838055, 0.1520071, -24.6532784, 0.1729176, -23.3904648, 23.3310623
12: -49.0657730, -8.5827894, -49.0472069, -8.5607967, -34.0702744, 34.0364914
13: -28.5310936, 19.1472130, -28.5436611, 19.1251411, -47.6562347, 47.6908722
14: -41.7726936, 8.8656769, -41.7954559, 8.8603649, -44.0218506, 43.9972000
15: -1.1683407, 28.0870152, -1.1987643, 28.0668659, -24.5986710, 24.6355209
16: -26.6427555, 10.5194664, -26.6397285, 10.5068226, -34.6554565, 34.6527710
17: -39.8273773, 4.8669853, -39.8282967, 4.8678713, -35.9473877, 35.9769821
18: -9.7291842, 26.5204620, -9.6845083, 26.5383434, -36.2675285, 36.2049713
19: -17.4056606, 9.2516346, -17.3857841, 9.2811508, -26.6868114, 26.6374187
20: -25.3983536, 3.6871696, -25.3788834, 3.7140198, -28.8029175, 28.7523422
21: -21.9893799, 8.0006752, -21.9645214, 8.0394135, -30.0287933, 29.9651966
22: -10.4333019, 21.2776318, -10.4127302, 21.2933674, -29.7483673, 29.7119751
23: -10.3214512, 18.0780258, -10.2929068, 18.0963783, -28.1753540, 28.1206741
24: -6.4599133, 22.3398247, -6.4386415, 22.3560352, -28.2236176, 28.1893387
25: -9.4903860, 22.7883301, -9.4708977, 22.8113747, -32.1954803, 32.1687164
26: -23.1739769, 24.3570709, -23.1336994, 24.3783760, -46.9485474, 46.8670807
27: -15.8851690, 17.3830223, -15.8756390, 17.4030151, -33.2881851, 33.2586594
28: -13.9567261, 19.3032532, -13.9252224, 19.3275394, -31.4981384, 31.4416656
29: -10.8952980, 15.4022417, -10.8769016, 15.4155579, -23.1047821, 23.0790482
30: -30.3314266, -0.1503301, -30.2986183, -0.1326282, -27.8165436, 27.7430420
31: -18.1533012, 10.6691504, -18.1380100, 10.7047281, -28.8580284, 28.8071594
32: -51.2068977, -16.1501713, -51.1933250, -16.1393337, -27.7833633, 27.7706375
33: -69.0467148, -12.0619450, -69.0484314, -12.0458174, -49.8896637, 49.8806763
34: -63.1077576, -21.4842606, -63.0784149, -21.4792900, -29.5760498, 29.5423660
35: -42.8081245, -0.5469568, -42.7919273, -0.5394015, -34.4035339, 34.4010239
36: -42.2407875, 2.7268076, -42.2266083, 2.7449131, -36.3126831, 36.2927017
37: -75.1976547, -19.0229530, -75.1796799, -19.0130177, -41.8601379, 41.8502045
38: -52.3303909, 1.9616623, -52.2980309, 1.9865994, -47.2350159, 47.1886749
39: -72.2401657, -13.6989307, -72.2413330, -13.6937828, -54.4369507, 54.4371490
40: -76.4356003, -36.7276611, -76.4335938, -36.7309837, -28.9432297, 28.9473572
41: -51.9936523, -11.2250862, -51.9814072, -11.2100220, -29.1632080, 29.1792603
42: -47.7678299, -16.2160378, -47.7547379, -16.2069740, -24.4907913, 24.4652100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9593688, upper bound: 15.0053831
time: 21.19 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9593688, upper bound: 15.0298095
time: 40.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.7562828, 27.0800591, -5.9166660, 27.0912094, -32.8474922, 32.9967270
1: 0.0642235, 22.9023800, -0.0283604, 22.9133472, -20.6348953, 20.7332535
2: -0.8485484, 25.0986347, -0.9601674, 25.1082039, -23.6119385, 23.7257767
3: -12.5735302, 15.9494572, -12.6576214, 15.9578991, -23.2665482, 23.3348007
4: -6.2164302, 20.8277473, -6.3471341, 20.8348732, -23.0020752, 23.1242218
5: -11.8643379, 20.9225502, -11.9551992, 20.9341183, -30.6142960, 30.6999741
6: -66.2093506, -30.7511787, -66.2186432, -30.6441364, -24.8632431, 24.7770119
7: -17.8371906, 14.9337254, -17.9193726, 14.9416485, -27.3749619, 27.4594879
8: -19.2432461, 10.2755260, -19.3801346, 10.2880936, -24.5700836, 24.6846619
9: -7.1503868, 22.8841496, -7.2569132, 22.8913879, -29.0587616, 29.1778946
10: -31.5928230, 12.2734165, -31.6455898, 12.2924299, -39.1281738, 39.1396561
11: -24.7015514, 0.1531932, -24.7224655, 0.2562568, -23.4892578, 23.3967209
12: -49.0748024, -8.5804119, -49.0847778, -8.4707251, -34.1700439, 34.0755310
13: -28.5337715, 19.1600037, -28.6202621, 19.1865788, -47.7203522, 47.7802658
14: -41.7748985, 8.8676357, -41.8544807, 8.8808451, -44.0498352, 44.0625763
15: -1.1712942, 28.0990486, -1.2971506, 28.1185589, -24.6453934, 24.7487144
16: -26.6469994, 10.5259781, -26.7063942, 10.5381241, -34.6927338, 34.7369232
17: -39.8319702, 4.8678942, -39.8793411, 4.9025917, -36.0020599, 36.0387268
18: -9.7510662, 26.5212746, -9.7647686, 26.6114426, -36.3625107, 36.2860413
19: -17.4165573, 9.2517242, -17.4330883, 9.3272247, -26.7437820, 26.6848125
20: -25.4108276, 3.6887634, -25.4327984, 3.7847764, -28.8916550, 28.8059540
21: -22.0037537, 8.0013742, -22.0279465, 8.1150885, -30.1188431, 30.0293198
22: -10.4435587, 21.2780762, -10.4594555, 21.3371410, -29.8287048, 29.7521362
23: -10.3362103, 18.0795250, -10.3541193, 18.1659050, -28.2723694, 28.1775970
24: -6.4728827, 22.3406639, -6.4930363, 22.3984222, -28.2855301, 28.2397614
25: -9.5015144, 22.7897491, -9.5242138, 22.8586540, -32.2863312, 32.2163010
26: -23.1952667, 24.3574409, -23.2159729, 24.4668198, -47.0655365, 46.9482269
27: -15.8888483, 17.3846989, -15.9015036, 17.4469128, -33.3357620, 33.2862015
28: -13.9733105, 19.3041210, -13.9933262, 19.4054451, -31.5940399, 31.5082626
29: -10.9044514, 15.4027081, -10.9170609, 15.4681396, -23.1719971, 23.1179199
30: -30.3504047, -0.1477754, -30.3714085, -0.0593371, -27.9117889, 27.8160248
31: -18.1609173, 10.6697989, -18.1855888, 10.7498236, -28.9107399, 28.8553886
32: -51.2157211, -16.1463184, -51.2330589, -16.0635681, -27.8709030, 27.8079147
33: -69.0492477, -12.0599070, -69.0747757, -12.0132170, -49.9266052, 49.9115753
34: -63.1241913, -21.4824867, -63.1433792, -21.3915691, -29.6844482, 29.6010361
35: -42.8168259, -0.5458121, -42.8311539, -0.4958262, -34.4538727, 34.4409103
36: -42.2489624, 2.7279365, -42.2598915, 2.8070865, -36.3840485, 36.3266144
37: -75.2080841, -19.0219688, -75.2235565, -18.9776039, -41.9076691, 41.8981705
38: -52.3478127, 1.9642353, -52.3715286, 2.0807014, -47.3478546, 47.2612305
39: -72.2429352, -13.6988335, -72.2672577, -13.6766453, -54.4658508, 54.4665375
40: -76.4394760, -36.7256660, -76.4603271, -36.7075653, -28.9706955, 28.9767532
41: -51.9998894, -11.2234716, -52.0098839, -11.1417561, -29.2393951, 29.2074203
42: -47.7755051, -16.2140293, -47.7856636, -16.1488400, -24.5694885, 24.4957008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9911798, upper bound: 15.0054839
time: 28.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9911798, upper bound: 15.0299183
time: 28.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.8339844, 27.1144295, -5.8237715, 27.0596085, -32.8935928, 32.9382019
1: 0.0201020, 22.9303093, 0.0258832, 22.8750000, -20.6448593, 20.6998978
2: -0.8962469, 25.1311378, -0.8957388, 25.0618362, -23.6117783, 23.6895370
3: -12.6261606, 15.9916859, -12.6249218, 15.9321404, -23.2880402, 23.3420944
4: -6.2760916, 20.8443089, -6.2691956, 20.7925949, -23.0151443, 23.0605164
5: -11.9036331, 20.9506645, -11.9014406, 20.8962822, -30.6182098, 30.6719437
6: -66.2526855, -30.6898422, -66.1928787, -30.6984558, -24.8427505, 24.8149071
7: -17.8753090, 14.9516602, -17.8708744, 14.9064817, -27.3784943, 27.4275284
8: -19.3150158, 10.3171463, -19.3151855, 10.2342196, -24.5838699, 24.6596832
9: -7.1910911, 22.9028320, -7.1778650, 22.8656502, -29.0785675, 29.1349411
10: -31.6126823, 12.3006802, -31.6006660, 12.2728672, -39.1266174, 39.1172485
11: -24.7310925, 0.1946783, -24.6559849, 0.1886785, -23.4583969, 23.3735352
12: -49.0847473, -8.5270777, -49.0488892, -8.5435047, -34.1100616, 34.0946884
13: -28.5671959, 19.1931534, -28.5561981, 19.1301079, -47.6973038, 47.7493515
14: -41.8263702, 8.9154072, -41.8131332, 8.8623724, -44.0838318, 44.0724258
15: -1.2354727, 28.1501350, -1.2234216, 28.0700912, -24.6657410, 24.7232399
16: -26.6689281, 10.5456495, -26.6445427, 10.5093899, -34.6946793, 34.7031631
17: -39.8681602, 4.9034538, -39.8419495, 4.8698444, -35.9866943, 36.0281448
18: -9.7936497, 26.5532036, -9.6866999, 26.5505028, -36.3441544, 36.2399025
19: -17.4756336, 9.2998762, -17.3879089, 9.3013878, -26.7770214, 26.6877861
20: -25.4795570, 3.7366250, -25.3820610, 3.7340689, -28.9033356, 28.8019791
21: -22.0766182, 8.0630550, -21.9679623, 8.0650558, -30.1416740, 30.0310173
22: -10.4815149, 21.3050079, -10.4151335, 21.3033390, -29.8114624, 29.7492065
23: -10.3787918, 18.1160793, -10.2953482, 18.1104774, -28.2489090, 28.1597214
24: -6.5223150, 22.3702984, -6.4416676, 22.3678455, -28.3038330, 28.2242966
25: -9.5602283, 22.8322144, -9.4745731, 22.8276405, -32.2854004, 32.2201233
26: -23.2410126, 24.3921032, -23.1370583, 24.3913002, -47.0359497, 46.9173279
27: -15.9331388, 17.4240265, -15.8784513, 17.4184456, -33.3515854, 33.3024788
28: -14.0221596, 19.3455734, -13.9279642, 19.3439465, -31.5792389, 31.4856796
29: -10.9163055, 15.4300785, -10.8780031, 15.4239616, -23.1354294, 23.1070938
30: -30.4018402, -0.1054771, -30.3013191, -0.1165171, -27.9036789, 27.7896576
31: -18.2305164, 10.7297421, -18.1416702, 10.7291470, -28.9596634, 28.8714123
32: -51.2388229, -16.1088581, -51.1956635, -16.1276054, -27.8276215, 27.8123322
33: -69.1004181, -12.0293970, -69.0535660, -12.0349979, -49.9531097, 49.9168091
34: -63.1440811, -21.4673309, -63.0814629, -21.4736996, -29.6244278, 29.5662460
35: -42.8463707, -0.5267539, -42.7947083, -0.5331235, -34.4509583, 34.4273300
36: -42.2846603, 2.7645853, -42.2283707, 2.7584639, -36.3732300, 36.3320694
37: -75.2371445, -19.0035286, -75.1823196, -19.0077190, -41.9075775, 41.8711700
38: -52.4318161, 2.0234714, -52.3029900, 2.0085874, -47.3576508, 47.2517700
39: -72.2929840, -13.6785669, -72.2460785, -13.6894474, -54.5118103, 54.4690399
40: -76.4586639, -36.7131042, -76.4383163, -36.7272339, -28.9687347, 28.9659424
41: -52.0209579, -11.1905336, -51.9829559, -11.1992073, -29.2043991, 29.2136993
42: -47.7886581, -16.1881142, -47.7561760, -16.1990166, -24.5192795, 24.4931107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0019284, upper bound: 15.0090110
time: 21.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0019284, upper bound: 15.0334459
time: 21.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.8372936, 27.1232738, -5.9481678, 27.0931625, -32.9304581, 33.0714417
1: 0.0181563, 22.9404049, -0.0458004, 22.9148064, -20.6816025, 20.7898483
2: -0.8968902, 25.1434116, -0.9800425, 25.1095352, -23.6605453, 23.7903824
3: -12.6269398, 15.9979677, -12.6792946, 15.9591475, -23.3177261, 23.4060593
4: -6.2777090, 20.8556538, -6.3700728, 20.8358955, -23.0624008, 23.1762772
5: -11.9045658, 20.9606819, -11.9705524, 20.9359989, -30.6587753, 30.7546692
6: -66.2573013, -30.6877041, -66.2202301, -30.6208344, -24.9368744, 24.8384857
7: -17.8766518, 14.9607115, -17.9343719, 14.9427071, -27.4174500, 27.5041122
8: -19.3155441, 10.3309517, -19.4096298, 10.2900276, -24.6396866, 24.7700653
9: -7.1935225, 22.9101791, -7.2711272, 22.8931026, -29.1075516, 29.2370300
10: -31.6154060, 12.3018017, -31.6505604, 12.2963896, -39.1596680, 39.1820831
11: -24.7488441, 0.1959035, -24.7251034, 0.2720370, -23.5572128, 23.4392128
12: -49.0937881, -8.5247326, -49.0864716, -8.4533510, -34.2098694, 34.1337128
13: -28.5699120, 19.2059727, -28.6327572, 19.1914406, -47.7613525, 47.8387299
14: -41.8286514, 8.9173679, -41.8720779, 8.8828926, -44.1118317, 44.1378937
15: -1.2383590, 28.1621857, -1.3218875, 28.1217289, -24.7124557, 24.8364258
16: -26.6732178, 10.5521727, -26.7112160, 10.5406485, -34.7318726, 34.7873230
17: -39.8727913, 4.9043951, -39.8929672, 4.9046350, -36.0414276, 36.0898743
18: -9.8155746, 26.5540848, -9.7670345, 26.6236210, -36.4391937, 36.3211212
19: -17.4865074, 9.3000393, -17.4352608, 9.3474407, -26.8339481, 26.7353001
20: -25.4920044, 3.7382343, -25.4359055, 3.8048341, -28.9920654, 28.8556061
21: -22.0910301, 8.0637808, -22.0313988, 8.1407509, -30.2317810, 30.0951805
22: -10.4918013, 21.3054066, -10.4618320, 21.3471069, -29.8917694, 29.7894516
23: -10.3936758, 18.1175442, -10.3565865, 18.1799831, -28.3459244, 28.2166367
24: -6.5352502, 22.3711433, -6.4960375, 22.4102688, -28.3657532, 28.2746887
25: -9.5712910, 22.8335686, -9.5278816, 22.8748856, -32.3762131, 32.2677155
26: -23.2621918, 24.3925152, -23.2193890, 24.4797344, -47.1529694, 46.9984894
27: -15.9368076, 17.4256649, -15.9042721, 17.4624329, -33.3992386, 33.3299370
28: -14.0386686, 19.3464031, -13.9960651, 19.4218273, -31.6751862, 31.5522461
29: -10.9254723, 15.4304867, -10.9181728, 15.4765692, -23.2026215, 23.1459808
30: -30.4208488, -0.1029449, -30.3741455, -0.0432358, -27.9989395, 27.8626404
31: -18.2381020, 10.7303686, -18.1892319, 10.7742424, -29.0123444, 28.9196014
32: -51.2476273, -16.1049957, -51.2353745, -16.0518303, -27.9151840, 27.8496094
33: -69.1029587, -12.0271940, -69.0800018, -12.0024090, -49.9901276, 49.9478302
34: -63.1604271, -21.4655685, -63.1464081, -21.3859272, -29.7328186, 29.6248779
35: -42.8550949, -0.5255773, -42.8339844, -0.4895856, -34.5013428, 34.4672623
36: -42.2928810, 2.7656460, -42.2616653, 2.8205504, -36.4446869, 36.3659668
37: -75.2475891, -19.0025063, -75.2261429, -18.9723110, -41.9551849, 41.9191132
38: -52.4492035, 2.0260901, -52.3764839, 2.1027155, -47.4704437, 47.3242340
39: -72.2957153, -13.6783276, -72.2720490, -13.6723270, -54.5407104, 54.4984741
40: -76.4624939, -36.7110176, -76.4650726, -36.7038460, -28.9962158, 28.9953537
41: -52.0271378, -11.1888905, -52.0114136, -11.1309414, -29.2806702, 29.2418365
42: -47.7963066, -16.1860504, -47.7870827, -16.1408653, -24.5979538, 24.5235825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=198, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1677

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0335311, upper bound: 15.0090859
time: 35.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0335311, upper bound: 15.0335312
time: 22.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 59.95 seconds
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9446164, upper bound: 15.0052804
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9446164, upper bound: 15.0297336
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9764498, upper bound: 15.0053759
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9764498, upper bound: 15.0298402
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9805841, upper bound: 15.0086862
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9805841, upper bound: 15.0331939
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0122627, upper bound: 15.0087678
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0122627, upper bound: 15.0332837
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0142094, upper bound: 14.9326440
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0142094, upper bound: 14.9566196
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9593688, upper bound: 15.0053831
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9593688, upper bound: 15.0298095
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9911798, upper bound: 15.0054839
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -14.9911798, upper bound: 15.0299183
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0019284, upper bound: 15.0090110
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0019284, upper bound: 15.0334459
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0335311, upper bound: 15.0090859
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 59.95
Output dim: 15, lower bound: -15.0335311, upper bound: 15.0335312

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.7515259, 27.0348625, -5.7377605, 27.0520821, -32.8036079, 32.7726212
1: 0.0165374, 22.8739777, 0.0665135, 22.8681450, -20.6421814, 20.5772018
2: -0.8516784, 25.0497742, -0.8427498, 25.0555096, -23.5736923, 23.5466080
3: -12.6107311, 15.9151287, -12.5748701, 15.9254122, -23.2718124, 23.2429047
4: -6.2527957, 20.8070755, -6.2169590, 20.7871647, -22.9960175, 22.9565506
5: -11.8931789, 20.8859997, -11.8618374, 20.8869095, -30.5944977, 30.5602493
6: -66.1840057, -30.7726784, -66.1857910, -30.7509899, -24.7076035, 24.7449188
7: -17.8903542, 14.9087572, -17.8363075, 14.8997421, -27.3848724, 27.3387833
8: -19.2717400, 10.2217121, -19.2396374, 10.2255650, -24.5605927, 24.5046730
9: -7.2114677, 22.8570633, -7.1494594, 22.8523598, -29.0999908, 29.0076065
10: -31.6000881, 12.2216644, -31.5836525, 12.2403641, -39.0698547, 38.8965073
11: -24.6603470, 0.1309810, -24.6457291, 0.1465423, -23.3335037, 23.3623123
12: -49.0561371, -8.5539951, -49.0401840, -8.5864391, -34.0180740, 34.0630798
13: -28.4736080, 19.0972252, -28.5087414, 19.1154442, -47.5890503, 47.6059647
14: -41.7306290, 8.8308306, -41.7643929, 8.8550072, -43.9748535, 43.8681030
15: -1.1441302, 28.0458488, -1.1557765, 28.0608864, -24.6038742, 24.5570526
16: -26.7002487, 10.5094414, -26.6318436, 10.4966402, -34.7054443, 34.5868912
17: -39.7518387, 4.8830099, -39.7752762, 4.8630047, -35.8222046, 35.9462433
18: -9.6595697, 26.5024567, -9.6753750, 26.5152683, -36.1748390, 36.1778336
19: -17.3629589, 9.2654114, -17.3796749, 9.2536440, -26.6166039, 26.6450863
20: -25.3374043, 3.6550980, -25.3695831, 3.6779511, -28.6883621, 28.7568970
21: -21.9245567, 7.9328332, -21.9556713, 7.9881377, -29.9126949, 29.8885040
22: -10.4149532, 21.3193741, -10.4057178, 21.2829456, -29.6829224, 29.7347260
23: -10.2897129, 18.1032486, -10.2865667, 18.0773182, -28.1219559, 28.1532288
24: -6.4061441, 22.3417320, -6.4272532, 22.3362732, -28.1489410, 28.1856155
25: -9.4488049, 22.8119965, -9.4599075, 22.7869015, -32.1080322, 32.1769867
26: -23.1314278, 24.3975220, -23.1230927, 24.3627853, -46.8340912, 46.8793335
27: -15.8585682, 17.3616695, -15.8677340, 17.3852825, -33.2438507, 33.2294044
28: -13.9194756, 19.3479004, -13.9161749, 19.3131561, -31.4201126, 31.4868088
29: -10.8934803, 15.4368725, -10.8702908, 15.4038363, -23.0706406, 23.1055679
30: -30.2757568, -0.1738632, -30.2922993, -0.1652994, -27.7169189, 27.7634735
31: -18.0973682, 10.6414318, -18.1284447, 10.6622458, -28.7596130, 28.7698765
32: -51.1869507, -16.1590652, -51.1862335, -16.1570206, -27.7370758, 27.7676926
33: -69.0170059, -12.0293865, -69.0347519, -12.0610819, -49.8307037, 49.9169617
34: -63.0681572, -21.4502335, -63.0677338, -21.4899006, -29.5382233, 29.5796967
35: -42.7849312, -0.5003633, -42.7811127, -0.5465508, -34.3347778, 34.4504700
36: -42.2230682, 2.7919812, -42.2174759, 2.7378082, -36.2219238, 36.3619995
37: -75.1612015, -18.9612770, -75.1638184, -19.0198307, -41.7810211, 41.9127274
38: -52.2779465, 2.0151129, -52.2808723, 1.9679270, -47.1307373, 47.2337036
39: -72.2070236, -13.6711750, -72.2278900, -13.7038918, -54.3807373, 54.4585114
40: -76.4050217, -36.7195587, -76.4203339, -36.7365112, -28.9405899, 28.9182129
41: -51.9832840, -11.2078428, -51.9759331, -11.2206459, -29.0870209, 29.1708984
42: -47.7621918, -16.2130890, -47.7502899, -16.2174149, -24.4724045, 24.5020027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9403291, upper bound: 15.0059371
time: 25.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9440285, upper bound: 15.0291943
time: 14.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.7549191, 27.0437088, -5.8620949, 27.0856647, -32.8405838, 32.9058037
1: 0.0145931, 22.8840866, -0.0051599, 22.9079628, -20.6789474, 20.6671219
2: -0.8523381, 25.0620499, -0.9270585, 25.1032143, -23.6224136, 23.6474228
3: -12.6115561, 15.9214096, -12.6292629, 15.9524174, -23.3014984, 23.3068695
4: -6.2544508, 20.8184624, -6.3177595, 20.8305130, -23.0432968, 23.0722656
5: -11.8940811, 20.8960419, -11.9309559, 20.9266281, -30.6350098, 30.6428833
6: -66.1885986, -30.7705269, -66.2131805, -30.6734905, -24.8016739, 24.7684937
7: -17.8916969, 14.9178057, -17.8997154, 14.9359550, -27.4238281, 27.4153824
8: -19.2722168, 10.2354860, -19.3341064, 10.2814064, -24.6164322, 24.6150360
9: -7.2139273, 22.8643818, -7.2427063, 22.8798103, -29.1289520, 29.1096649
10: -31.6028557, 12.2228165, -31.6333961, 12.2637939, -39.1029739, 38.9612885
11: -24.6780434, 0.1322165, -24.7148228, 0.2298574, -23.4322968, 23.4279900
12: -49.0651779, -8.5516577, -49.0778122, -8.4962759, -34.1177216, 34.1021271
13: -28.4762936, 19.1099663, -28.5853024, 19.1768608, -47.6531525, 47.6952667
14: -41.7329674, 8.8328714, -41.8233566, 8.8755741, -44.0028687, 43.9334564
15: -1.1470737, 28.0579147, -1.2541790, 28.1125908, -24.6506424, 24.6702080
16: -26.7045269, 10.5159454, -26.6984730, 10.5279179, -34.7426376, 34.6710663
17: -39.7565804, 4.8839283, -39.8263969, 4.8976974, -35.8769226, 36.0079651
18: -9.6814690, 26.5033112, -9.7556992, 26.5883617, -36.2698288, 36.2590103
19: -17.3737907, 9.2655039, -17.4269753, 9.2996874, -26.6734772, 26.6924782
20: -25.3498421, 3.6567051, -25.4234657, 3.7487161, -28.7770691, 28.8105316
21: -21.9389191, 7.9335723, -22.0190868, 8.0637808, -30.0027008, 29.9526596
22: -10.4252110, 21.3198242, -10.4524174, 21.3266964, -29.7632370, 29.7748947
23: -10.3045464, 18.1047764, -10.3477812, 18.1468468, -28.2189331, 28.2102127
24: -6.4190946, 22.3425617, -6.4816566, 22.3786907, -28.2108994, 28.2360535
25: -9.4598732, 22.8133812, -9.5132084, 22.8341370, -32.1988525, 32.2246094
26: -23.1526852, 24.3978462, -23.2054253, 24.4512558, -46.9510803, 46.9606781
27: -15.8622684, 17.3633595, -15.8936377, 17.4291458, -33.2914124, 33.2569962
28: -13.9360409, 19.3487263, -13.9843273, 19.3910751, -31.5160217, 31.5534439
29: -10.9025764, 15.4373026, -10.9104843, 15.4564438, -23.1378250, 23.1444626
30: -30.2947197, -0.1712892, -30.3651123, -0.0920663, -27.8120956, 27.8364792
31: -18.1049252, 10.6421089, -18.1760139, 10.7073326, -28.8122578, 28.8181229
32: -51.1958237, -16.1551895, -51.2259331, -16.0812836, -27.8245850, 27.8049774
33: -69.0195007, -12.0272827, -69.0611115, -12.0285845, -49.8676910, 49.9479218
34: -63.0845642, -21.4484253, -63.1328278, -21.4021301, -29.6465149, 29.6383591
35: -42.7937126, -0.4992800, -42.8203621, -0.5030308, -34.3850861, 34.4904480
36: -42.2313538, 2.7930696, -42.2507706, 2.7999337, -36.2932892, 36.3959122
37: -75.1716156, -18.9603539, -75.2076797, -18.9844799, -41.8285828, 41.9606400
38: -52.2954178, 2.0177274, -52.3544121, 2.0619483, -47.2436066, 47.3062286
39: -72.2097626, -13.6710892, -72.2539520, -13.6867886, -54.4096069, 54.4880676
40: -76.4089203, -36.7175331, -76.4470520, -36.7131462, -28.9680786, 28.9475784
41: -51.9894905, -11.2062321, -52.0043907, -11.1524143, -29.1631927, 29.1990509
42: -47.7698669, -16.2110233, -47.7811813, -16.1593742, -24.5510483, 24.5324631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9721178, upper bound: 15.0060594
time: 39.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9758613, upper bound: 15.0292965
time: 36.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.8323383, 27.0781765, -5.7690639, 27.0540504, -32.8863907, 32.8472404
1: -0.0294499, 22.9120522, 0.0491529, 22.8696194, -20.6887741, 20.6337433
2: -0.8998482, 25.0945835, -0.8626027, 25.0568161, -23.6218185, 23.6111450
3: -12.6639309, 15.9636774, -12.5965157, 15.9266539, -23.3228073, 23.3141479
4: -6.3133960, 20.8350163, -6.2395339, 20.7881813, -23.0560608, 23.0083733
5: -11.9331350, 20.9239330, -11.8771038, 20.8887844, -30.6384964, 30.6148834
6: -66.2322006, -30.7094212, -66.1873169, -30.7278404, -24.7811050, 24.8060341
7: -17.9295731, 14.9357071, -17.8512383, 14.9007864, -27.4270248, 27.3833466
8: -19.3427124, 10.2767086, -19.2685204, 10.2274694, -24.6299362, 24.5898895
9: -7.2553725, 22.8856697, -7.1636143, 22.8540554, -29.1485214, 29.0665970
10: -31.6263313, 12.2523327, -31.5885677, 12.2442808, -39.1057129, 38.9406738
11: -24.7075005, 0.1734567, -24.6483803, 0.1622393, -23.4012756, 23.4046059
12: -49.0747452, -8.4992085, -49.0418549, -8.5692673, -34.0561371, 34.1201172
13: -28.5092239, 19.1437111, -28.5211143, 19.1203384, -47.6295624, 47.6648254
14: -41.7839890, 8.8808002, -41.7819977, 8.8571186, -44.0357056, 43.9423904
15: -1.2111273, 28.1091404, -1.1803727, 28.0641041, -24.6704941, 24.6448441
16: -26.7260857, 10.5360241, -26.6366215, 10.4991617, -34.7447205, 34.6366501
17: -39.7921600, 4.9191513, -39.7888756, 4.8649769, -35.8598404, 35.9948654
18: -9.7240515, 26.5351486, -9.6776066, 26.5273933, -36.2514458, 36.2127533
19: -17.4330158, 9.3135643, -17.3818340, 9.2737761, -26.7067909, 26.6953983
20: -25.4185314, 3.7044349, -25.3727074, 3.6978981, -28.7886810, 28.8062592
21: -22.0118065, 7.9950733, -21.9591007, 8.0137739, -30.0255814, 29.9541740
22: -10.4631214, 21.3465786, -10.4081573, 21.2928696, -29.7459259, 29.7718582
23: -10.3472347, 18.1410751, -10.2889786, 18.0913391, -28.1955872, 28.1921768
24: -6.4685268, 22.3722076, -6.4302511, 22.3480110, -28.2290649, 28.2204895
25: -9.5185642, 22.8557129, -9.4635811, 22.8030224, -32.1977615, 32.2281723
26: -23.1983662, 24.4323139, -23.1264687, 24.3756351, -46.9214630, 46.9291992
27: -15.9067707, 17.4024200, -15.8705864, 17.4006386, -33.3074112, 33.2730064
28: -13.9850216, 19.3899765, -13.9189243, 19.3294582, -31.5013809, 31.5304489
29: -10.9161644, 15.4654379, -10.8713951, 15.4121323, -23.1020432, 23.1332703
30: -30.3461056, -0.1292195, -30.2949867, -0.1492889, -27.8038330, 27.8098907
31: -18.1745453, 10.7019176, -18.1320610, 10.6866150, -28.8611603, 28.8339787
32: -51.2184525, -16.1178398, -51.1885834, -16.1453114, -27.7807007, 27.8091583
33: -69.0709457, -11.9974070, -69.0399628, -12.0503531, -49.8945160, 49.9524231
34: -63.1044388, -21.4334164, -63.0708656, -21.4843254, -29.5866852, 29.6033859
35: -42.8231277, -0.4805663, -42.7838669, -0.5403156, -34.3824310, 34.4763641
36: -42.2675705, 2.8292074, -42.2191925, 2.7512257, -36.2828064, 36.4009857
37: -75.2016602, -18.9409199, -75.1663361, -19.0145302, -41.8280792, 41.9342499
38: -52.3795204, 2.0768261, -52.2858429, 1.9897957, -47.2533264, 47.2965546
39: -72.2600403, -13.6510916, -72.2327499, -13.6995907, -54.4541626, 54.4892426
40: -76.4277649, -36.7051659, -76.4250336, -36.7328224, -28.9658585, 28.9365921
41: -52.0106735, -11.1733999, -51.9774742, -11.2098255, -29.1299438, 29.2053833
42: -47.7829170, -16.1853638, -47.7517433, -16.2095222, -24.5006638, 24.5292854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9762057, upper bound: 15.0091787
time: 29.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9799717, upper bound: 15.0326491
time: 26.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.8357182, 27.0870438, -5.8935022, 27.0876637, -32.9233818, 32.9805450
1: -0.0313861, 22.9221668, -0.0225410, 22.9094353, -20.7255249, 20.7237091
2: -0.9005072, 25.1068287, -0.9468544, 25.1045132, -23.6705856, 23.7120056
3: -12.6647396, 15.9699516, -12.6508560, 15.9537144, -23.3525085, 23.3780746
4: -6.3150339, 20.8463688, -6.3403416, 20.8315334, -23.1033554, 23.1241341
5: -11.9340811, 20.9338989, -11.9462013, 20.9284630, -30.6790466, 30.6975403
6: -66.2368164, -30.7072754, -66.2146683, -30.6502762, -24.8751678, 24.8295937
7: -17.9308891, 14.9447517, -17.9146080, 14.9370136, -27.4659576, 27.4599380
8: -19.3432484, 10.2905016, -19.3629189, 10.2833099, -24.6857986, 24.7002678
9: -7.2577629, 22.8930569, -7.2568479, 22.8815136, -29.1774521, 29.1687012
10: -31.6290894, 12.2534122, -31.6383438, 12.2676840, -39.1387711, 39.0054626
11: -24.7252502, 0.1747105, -24.7174549, 0.2455878, -23.5001144, 23.4702682
12: -49.0838432, -8.4968567, -49.0795174, -8.4791813, -34.1558075, 34.1592178
13: -28.5118942, 19.1565056, -28.5976982, 19.1817932, -47.6936874, 47.7542038
14: -41.7862511, 8.8827782, -41.8409309, 8.8776569, -44.0637054, 44.0077820
15: -1.2140946, 28.1211796, -1.2787628, 28.1157646, -24.7172318, 24.7579651
16: -26.7302475, 10.5425606, -26.7033215, 10.5303974, -34.7819214, 34.7208405
17: -39.7967567, 4.9200854, -39.8399849, 4.8997278, -35.9145355, 36.0566101
18: -9.7460241, 26.5359955, -9.7579079, 26.6005096, -36.3465347, 36.2939034
19: -17.4439087, 9.3136711, -17.4291496, 9.3197975, -26.7637062, 26.7428207
20: -25.4310226, 3.7060211, -25.4266052, 3.7686598, -28.8774185, 28.8599167
21: -22.0261612, 7.9958549, -22.0225639, 8.0893888, -30.1155510, 30.0184193
22: -10.4733601, 21.3470097, -10.4548492, 21.3366203, -29.8262482, 29.8120651
23: -10.3620281, 18.1425915, -10.3502579, 18.1608524, -28.2926025, 28.2491074
24: -6.4814410, 22.3730183, -6.4846807, 22.3904438, -28.2909851, 28.2709808
25: -9.5296164, 22.8571033, -9.5169144, 22.8502541, -32.2886658, 32.2758255
26: -23.2195435, 24.4326496, -23.2088261, 24.4640865, -47.0385742, 47.0104218
27: -15.9104652, 17.4040833, -15.8964157, 17.4445667, -33.3550339, 33.3004990
28: -14.0015516, 19.3908138, -13.9870481, 19.4073505, -31.5972748, 31.5970230
29: -10.9253139, 15.4658308, -10.9115524, 15.4647455, -23.1692047, 23.1721420
30: -30.3650551, -0.1266294, -30.3677902, -0.0760176, -27.8990707, 27.8828888
31: -18.1821518, 10.7025280, -18.1796455, 10.7317162, -28.9138680, 28.8821735
32: -51.2272568, -16.1140251, -51.2282867, -16.0695667, -27.8681870, 27.8464203
33: -69.0734634, -11.9951754, -69.0663834, -12.0177803, -49.9314117, 49.9833832
34: -63.1208649, -21.4317093, -63.1357994, -21.3965664, -29.6950302, 29.6620865
35: -42.8318596, -0.4794450, -42.8230782, -0.4967949, -34.4328156, 34.5162582
36: -42.2758102, 2.8303435, -42.2524719, 2.8133461, -36.3541718, 36.4348450
37: -75.2121277, -18.9399757, -75.2102737, -18.9790668, -41.8755341, 41.9822311
38: -52.3968887, 2.0793123, -52.3593788, 2.0838223, -47.3661804, 47.3690491
39: -72.2627792, -13.6509075, -72.2586365, -13.6825190, -54.4830933, 54.5185852
40: -76.4317169, -36.7031708, -76.4518280, -36.7093811, -28.9933777, 28.9659882
41: -52.0168839, -11.1717911, -52.0059509, -11.1415749, -29.2062149, 29.2334824
42: -47.7905807, -16.1833572, -47.7826538, -16.1514263, -24.5792770, 24.5597229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -15.0079195, upper bound: 15.0092852
time: 23.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0116703, upper bound: 15.0327333
time: 23.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.8655214, 27.0743446, -5.7914705, 27.0559959, -32.9215164, 32.8658142
1: -0.0343812, 22.8925419, 0.0438950, 22.8713360, -20.7003708, 20.6338882
2: -0.9193556, 25.0873642, -0.8756285, 25.0584183, -23.6330795, 23.6186295
3: -12.6686125, 15.9421043, -12.6030903, 15.9288311, -23.3321304, 23.2511292
4: -6.3144641, 20.8155441, -6.2456951, 20.7891216, -23.0559006, 22.9979973
5: -11.9434500, 20.9112988, -11.8859062, 20.8912945, -30.6519699, 30.6055679
6: -66.2082367, -30.7119617, -66.1893539, -30.7224007, -24.7712097, 24.8000526
7: -17.9361839, 14.9226017, -17.8558083, 14.9025249, -27.4408112, 27.3613052
8: -19.3636360, 10.2689886, -19.2854767, 10.2307653, -24.6372299, 24.5621109
9: -7.2461996, 22.8760872, -7.1626258, 22.8605595, -29.1392059, 29.0637589
10: -31.6634521, 12.2793913, -31.5954552, 12.2661610, -39.1676254, 39.0733337
11: -24.7066307, 0.1851287, -24.6495037, 0.1722686, -23.3965378, 23.3868294
12: -49.0704269, -8.4922819, -49.0446091, -8.5612516, -34.0587158, 34.1297760
13: -28.5386200, 19.1710663, -28.5400505, 19.1248379, -47.6634598, 47.7111168
14: -41.7975922, 8.8724804, -41.7941895, 8.8591003, -44.0433807, 44.0044174
15: -1.2355409, 28.0971642, -1.1982565, 28.0660973, -24.6693039, 24.6406021
16: -26.7342186, 10.5215578, -26.6384087, 10.5021152, -34.7516022, 34.6502838
17: -39.8607712, 4.9533987, -39.8238297, 4.8673887, -35.9636383, 36.0716629
18: -9.7341089, 26.5488911, -9.6817913, 26.5380020, -36.2721100, 36.2306824
19: -17.4138680, 9.3200064, -17.3840218, 9.2811108, -26.6949787, 26.7040291
20: -25.4041672, 3.7269199, -25.3748188, 3.7134414, -28.7910309, 28.8141327
21: -22.0060730, 8.0345345, -21.9604568, 8.0391731, -30.0452461, 29.9949913
22: -10.4454985, 21.3412266, -10.4102592, 21.2930946, -29.7504578, 29.7747040
23: -10.3292236, 18.1433849, -10.2911863, 18.0958328, -28.1786499, 28.1845551
24: -6.4592633, 22.3808041, -6.4339557, 22.3557129, -28.2221527, 28.2331314
25: -9.4947290, 22.8620129, -9.4660730, 22.8109875, -32.1940994, 32.2426147
26: -23.1796036, 24.4285889, -23.1307487, 24.3781071, -46.9512329, 46.9365387
27: -15.8985109, 17.3972855, -15.8752737, 17.4024544, -33.3009644, 33.2725601
28: -13.9603148, 19.3780670, -13.9221535, 19.3274212, -31.4914703, 31.5138931
29: -10.9086676, 15.4667845, -10.8744020, 15.4152546, -23.1068497, 23.1402283
30: -30.3429375, -0.1046650, -30.2960777, -0.1330724, -27.8155670, 27.7976379
31: -18.1653061, 10.7257862, -18.1352692, 10.7045279, -28.8698349, 28.8610554
32: -51.2131386, -16.1166153, -51.1921768, -16.1404152, -27.7872162, 27.8061218
33: -69.0507431, -11.9934807, -69.0428543, -12.0461607, -49.8885040, 49.9575500
34: -63.1087379, -21.4239044, -63.0749283, -21.4797459, -29.5635147, 29.6110077
35: -42.8029785, -0.4797668, -42.7860794, -0.5394292, -34.3860474, 34.4814682
36: -42.2365646, 2.8110814, -42.2222176, 2.7447565, -36.2909088, 36.3865509
37: -75.1873093, -18.9408569, -75.1718597, -19.0133991, -41.8358765, 41.9371948
38: -52.3312416, 2.0605288, -52.2922707, 1.9857321, -47.2211761, 47.2916565
39: -72.2394943, -13.6412144, -72.2364502, -13.6940479, -54.4301758, 54.4961853
40: -76.4420929, -36.6949272, -76.4327927, -36.7312813, -28.9743881, 28.9520187
41: -51.9959679, -11.1811638, -51.9796906, -11.2104797, -29.1736221, 29.2130966
42: -47.7745476, -16.1869278, -47.7530479, -16.2075062, -24.4917526, 24.5006065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9552397, upper bound: 15.0060213
time: 29.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9587635, upper bound: 15.0292695
time: 17.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.8688431, 27.0831661, -5.9158058, 27.0895805, -32.9584236, 32.9989700
1: -0.0363295, 22.9026756, -0.0278344, 22.9111366, -20.7371368, 20.7238464
2: -0.9200029, 25.0996475, -0.9599087, 25.1061459, -23.6818237, 23.7195129
3: -12.6694431, 15.9483585, -12.6575184, 15.9558678, -23.3618164, 23.3150406
4: -6.3161101, 20.8269119, -6.3465500, 20.8324623, -23.1031723, 23.1137619
5: -11.9444160, 20.9213104, -11.9549971, 20.9309998, -30.6924591, 30.6882172
6: -66.2128448, -30.7098217, -66.2167511, -30.6448326, -24.8653107, 24.8236542
7: -17.9375572, 14.9316788, -17.9192562, 14.9387197, -27.4797440, 27.4379120
8: -19.3641357, 10.2827864, -19.3799458, 10.2866039, -24.6930695, 24.6724663
9: -7.2486434, 22.8834286, -7.2557998, 22.8880405, -29.1681366, 29.1658325
10: -31.6661797, 12.2805290, -31.6452446, 12.2897396, -39.2007904, 39.1381302
11: -24.7243023, 0.1863537, -24.7186050, 0.2556069, -23.4953995, 23.4524994
12: -49.0794754, -8.4899387, -49.0822639, -8.4712534, -34.1584702, 34.1687698
13: -28.5413170, 19.1838379, -28.6166992, 19.1862526, -47.7275696, 47.8005371
14: -41.7998734, 8.8745537, -41.8532295, 8.8796425, -44.0713501, 44.0697479
15: -1.2384949, 28.1092377, -1.2966342, 28.1177406, -24.7160416, 24.7538033
16: -26.7385025, 10.5280838, -26.7050476, 10.5334110, -34.7888641, 34.7344360
17: -39.8653870, 4.9543266, -39.8749161, 4.9021382, -36.0183563, 36.1334076
18: -9.7560797, 26.5497265, -9.7620592, 26.6111031, -36.3671837, 36.3117867
19: -17.4247665, 9.3200836, -17.4313526, 9.3271704, -26.7519379, 26.7514362
20: -25.4166489, 3.7284751, -25.4287300, 3.7842124, -28.8797150, 28.8676987
21: -22.0204468, 8.0353003, -22.0238914, 8.1149101, -30.1353569, 30.0591927
22: -10.4557638, 21.3416443, -10.4569998, 21.3368683, -29.8308258, 29.8149185
23: -10.3440619, 18.1448555, -10.3524561, 18.1653824, -28.2756042, 28.2415390
24: -6.4721465, 22.3815975, -6.4883647, 22.3980942, -28.2841492, 28.2835464
25: -9.5057507, 22.8633919, -9.5193872, 22.8582077, -32.2848892, 32.2902374
26: -23.2007065, 24.4290218, -23.2131348, 24.4665623, -47.0681763, 47.0177460
27: -15.9022188, 17.3989658, -15.9011011, 17.4463482, -33.3485680, 33.3000679
28: -13.9768190, 19.3789196, -13.9903021, 19.4053307, -31.5874176, 31.5805588
29: -10.9177809, 15.4672127, -10.9145889, 15.4678659, -23.1740036, 23.1790771
30: -30.3618565, -0.1021190, -30.3688030, -0.0597928, -27.9108353, 27.8706436
31: -18.1728916, 10.7263536, -18.1828003, 10.7495480, -28.9224396, 28.9091530
32: -51.2219849, -16.1128120, -51.2318840, -16.0646725, -27.8747635, 27.8434067
33: -69.0532837, -11.9913034, -69.0692444, -12.0135775, -49.9254303, 49.9885101
34: -63.1251755, -21.4221458, -63.1399307, -21.3919182, -29.6718826, 29.6696472
35: -42.8116989, -0.4786696, -42.8253441, -0.4958563, -34.4363708, 34.5214005
36: -42.2448235, 2.8122196, -42.2555466, 2.8068817, -36.3622894, 36.4204178
37: -75.1977158, -18.9398804, -75.2157211, -18.9779453, -41.8833771, 41.9851074
38: -52.3486519, 2.0631781, -52.3657532, 2.0798221, -47.3340607, 47.3642273
39: -72.2421112, -13.6410360, -72.2623825, -13.6768894, -54.4590149, 54.5256805
40: -76.4459686, -36.6928482, -76.4595642, -36.7079010, -29.0019073, 28.9814224
41: -52.0021591, -11.1795082, -52.0081177, -11.1422005, -29.2498169, 29.2412415
42: -47.7822495, -16.1848774, -47.7839584, -16.1493607, -24.5704193, 24.5310745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9870172, upper bound: 15.0061430
time: 30.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -14.9905808, upper bound: 15.0293759
time: 26.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.9465847, 27.1175537, -5.8228359, 27.0579834, -33.0045700, 32.9403915
1: -0.0804534, 22.9305534, 0.0264499, 22.8727970, -20.7470779, 20.6904831
2: -0.9677227, 25.1321564, -0.8955202, 25.0597305, -23.6816864, 23.6832352
3: -12.7220688, 15.9906254, -12.6248274, 15.9301014, -23.3833542, 23.3223572
4: -6.3758273, 20.8434715, -6.2686520, 20.7901344, -23.1162567, 23.0500717
5: -11.9837389, 20.9494324, -11.9012623, 20.8931713, -30.6964188, 30.6602402
6: -66.2561646, -30.6484833, -66.1909485, -30.6991348, -24.8448257, 24.8615570
7: -17.9756126, 14.9495487, -17.8707600, 14.9035263, -27.4832916, 27.4059296
8: -19.4359818, 10.3244038, -19.3150024, 10.2327070, -24.7068787, 24.6474571
9: -7.2893758, 22.9020538, -7.1767187, 22.8622684, -29.1879578, 29.1229324
10: -31.6860657, 12.3076954, -31.6003189, 12.2700796, -39.1992493, 39.1157913
11: -24.7538815, 0.2277899, -24.6521683, 0.1880598, -23.4645004, 23.4293060
12: -49.0894089, -8.4365444, -49.0462570, -8.5439615, -34.0984955, 34.1880188
13: -28.5749283, 19.2169304, -28.5525780, 19.1297340, -47.7046623, 47.7695084
14: -41.8513298, 8.9222336, -41.8118439, 8.8611231, -44.1053772, 44.0796280
15: -1.3027320, 28.1602364, -1.2229133, 28.0693169, -24.7363663, 24.7282982
16: -26.7604160, 10.5477142, -26.6432304, 10.5046272, -34.7907867, 34.7007751
17: -39.9015999, 4.9898429, -39.8375168, 4.8694668, -36.0029907, 36.1228561
18: -9.7986202, 26.5817509, -9.6840334, 26.5501900, -36.3488083, 36.2657852
19: -17.4838295, 9.3683071, -17.3861351, 9.3013620, -26.7851906, 26.7544422
20: -25.4853210, 3.7763596, -25.3779926, 3.7334750, -28.8914185, 28.8637924
21: -22.0933323, 8.0969944, -21.9638977, 8.0648308, -30.1581631, 30.0608921
22: -10.4936676, 21.3686275, -10.4126883, 21.3030796, -29.8134613, 29.8119812
23: -10.3866329, 18.1813927, -10.2936640, 18.1099224, -28.2521744, 28.2236404
24: -6.5216341, 22.4112701, -6.4369903, 22.3675003, -28.3023758, 28.2680969
25: -9.5644512, 22.9058838, -9.4697676, 22.8272228, -32.2839737, 32.2940369
26: -23.2464523, 24.4637833, -23.1341248, 24.3909798, -47.0386047, 46.9867554
27: -15.9464798, 17.4382935, -15.8780537, 17.4178638, -33.3643417, 33.3163452
28: -14.0256691, 19.4204044, -13.9249115, 19.3437805, -31.5725937, 31.5579376
29: -10.9296398, 15.4945917, -10.8755245, 15.4237070, -23.1374817, 23.1682739
30: -30.4132843, -0.0598109, -30.2987671, -0.1169541, -27.9026718, 27.8442764
31: -18.2424889, 10.7863817, -18.1388721, 10.7288532, -28.9713421, 28.9252548
32: -51.2450180, -16.0753098, -51.1945419, -16.1287537, -27.8314514, 27.8479614
33: -69.1043472, -11.9608555, -69.0479965, -12.0353622, -49.9518280, 49.9936829
34: -63.1450157, -21.4068775, -63.0779572, -21.4741230, -29.6118393, 29.6349030
35: -42.8412361, -0.4595726, -42.7888374, -0.5331850, -34.4335175, 34.5077820
36: -42.2804031, 2.8488150, -42.2238998, 2.7582335, -36.3514404, 36.4259644
37: -75.2267532, -18.9214535, -75.1743774, -19.0079880, -41.8832550, 41.9581833
38: -52.4326668, 2.1222978, -52.2972870, 2.0077281, -47.3438263, 47.3548126
39: -72.2921677, -13.6207771, -72.2411957, -13.6897173, -54.5049286, 54.5281372
40: -76.4651489, -36.6803436, -76.4375305, -36.7275238, -28.9998932, 28.9706116
41: -52.0232468, -11.1465378, -51.9812012, -11.1996193, -29.2148285, 29.2476730
42: -47.7953529, -16.1589584, -47.7544785, -16.1995239, -24.5202332, 24.5285492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9973826, upper bound: 15.0093171
time: 25.25 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0013553, upper bound: 15.0328687
time: 30.23 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.8323650, 27.0938148, -5.9461136, 27.0808849, -32.9132500, 33.0399284
1: 0.0204990, 22.9003201, -0.0447693, 22.8979168, -20.6620178, 20.7475739
2: -0.8949680, 25.1089401, -0.9791589, 25.0951233, -23.6440353, 23.7550049
3: -12.6255121, 15.9522266, -12.6786432, 15.9400606, -23.2979050, 23.3635330
4: -6.2744398, 20.8117085, -6.3686814, 20.8174973, -23.0411606, 23.1310959
5: -11.9029741, 20.9196911, -11.9698658, 20.9187584, -30.6398468, 30.7138443
6: -66.2439346, -30.6923790, -66.2146225, -30.6228180, -24.9215546, 24.8287697
7: -17.8751202, 14.9158058, -17.9336739, 14.9231930, -27.3951263, 27.4542770
8: -19.3134384, 10.2875338, -19.4087448, 10.2710857, -24.6176758, 24.7243118
9: -7.1881480, 22.8586864, -7.2688460, 22.8712311, -29.0797272, 29.1778488
10: -31.6113415, 12.2677689, -31.6488113, 12.2820616, -39.1405029, 39.1436920
11: -24.7352314, 0.1927602, -24.7192993, 0.2707362, -23.5274429, 23.4243317
12: -49.0535965, -8.5283670, -49.0691452, -8.4548454, -34.1669693, 34.1111755
13: -28.5617771, 19.2035179, -28.6293812, 19.1903896, -47.7521667, 47.8329010
14: -41.8199005, 8.9098339, -41.8683357, 8.8796968, -44.1003876, 44.1282501
15: -1.2341905, 28.1349373, -1.3200893, 28.1102180, -24.6977005, 24.8084450
16: -26.6688995, 10.4990940, -26.7094078, 10.5175438, -34.7025452, 34.7269897
17: -39.8074722, 4.9030948, -39.8650131, 4.9040689, -35.9688873, 36.0582047
18: -9.8087053, 26.5515862, -9.7640944, 26.6225395, -36.4312439, 36.3156815
19: -17.4432449, 9.2995138, -17.4171181, 9.3472519, -26.7904968, 26.7166328
20: -25.4702377, 3.7361500, -25.4263992, 3.8039763, -28.9524994, 28.8366241
21: -22.0781822, 8.0620642, -22.0259705, 8.1400328, -30.2182159, 30.0880356
22: -10.4505768, 21.3043385, -10.4438314, 21.3466339, -29.8479156, 29.7700653
23: -10.3536549, 18.1148968, -10.3398666, 18.1788826, -28.3050308, 28.1976471
24: -6.5076823, 22.3694649, -6.4842920, 22.4095345, -28.3342819, 28.2595978
25: -9.5259819, 22.8313103, -9.5083551, 22.8739414, -32.3269196, 32.2448578
26: -23.2197266, 24.3911877, -23.2015228, 24.4791527, -47.1093597, 46.9788818
27: -15.9337349, 17.4231052, -15.9029779, 17.4613228, -33.3950577, 33.3260841
28: -13.9925995, 19.3453026, -13.9767818, 19.4213448, -31.6285706, 31.5318298
29: -10.8846416, 15.4292698, -10.9007416, 15.4761066, -23.1607590, 23.1271057
30: -30.4049759, -0.1056566, -30.3672638, -0.0443985, -27.9734879, 27.8490143
31: -18.2133389, 10.7280579, -18.1787720, 10.7732372, -28.9865761, 28.9068298
32: -51.2385712, -16.1093731, -51.2315445, -16.0537109, -27.9041672, 27.8416977
33: -69.0624390, -12.0308380, -69.0629730, -12.0040236, -49.9446259, 49.9257812
34: -63.1307678, -21.4668446, -63.1338806, -21.3864689, -29.6990204, 29.6091919
35: -42.8119354, -0.5263488, -42.8158722, -0.4898708, -34.4546509, 34.4470291
36: -42.2455750, 2.7637630, -42.2417030, 2.8197565, -36.3935852, 36.3422546
37: -75.1852417, -19.0054188, -75.2001801, -18.9735661, -41.8971405, 41.8880234
38: -52.3856964, 2.0231791, -52.3489876, 2.1014028, -47.4011078, 47.2922363
39: -72.2578430, -13.6802998, -72.2561951, -13.6731462, -54.4999847, 54.4797821
40: -76.4593430, -36.7150078, -76.4637222, -36.7055359, -28.9814377, 28.9760056
41: -52.0107002, -11.1913471, -52.0043755, -11.1320028, -29.2630692, 29.2300339
42: -47.7811852, -16.1880417, -47.7807312, -16.1416779, -24.5788269, 24.5139389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0290262, upper bound: 14.9850822
time: 23.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0329532, upper bound: 15.0085055
time: 24.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.9499435, 27.1263657, -5.9472399, 27.0915508, -33.0414963, 33.0736046
1: -0.0824044, 22.9406910, -0.0451963, 22.9126015, -20.7838364, 20.7804337
2: -0.9683630, 25.1444321, -0.9797833, 25.1074352, -23.7304153, 23.7840729
3: -12.7228613, 15.9968910, -12.6791286, 15.9571609, -23.4130325, 23.3863220
4: -6.3774252, 20.8548050, -6.3694458, 20.8334637, -23.1635208, 23.1658745
5: -11.9846792, 20.9593925, -11.9703627, 20.9328003, -30.7368546, 30.7428513
6: -66.2607880, -30.6463165, -66.2183075, -30.6215305, -24.9389496, 24.8851166
7: -17.9769573, 14.9586372, -17.9341469, 14.9397917, -27.5222549, 27.4825821
8: -19.4364891, 10.3381987, -19.4094353, 10.2885284, -24.7627182, 24.7578354
9: -7.2917795, 22.9094315, -7.2699776, 22.8896980, -29.2169189, 29.2249527
10: -31.6887512, 12.3088417, -31.6501808, 12.2936287, -39.2322769, 39.1806107
11: -24.7716045, 0.2290542, -24.7212639, 0.2713697, -23.5633621, 23.4950027
12: -49.0984688, -8.4341946, -49.0839310, -8.4538221, -34.1983032, 34.2269897
13: -28.5776176, 19.2297363, -28.6292095, 19.1911087, -47.7687263, 47.8589478
14: -41.8536377, 8.9242268, -41.8709183, 8.8816528, -44.1333771, 44.1451263
15: -1.3056145, 28.1722832, -1.3213687, 28.1209297, -24.7830811, 24.8414841
16: -26.7646942, 10.5542555, -26.7098808, 10.5358982, -34.8280106, 34.7848587
17: -39.9062233, 4.9907589, -39.8885651, 4.9041815, -36.0576553, 36.1845551
18: -9.8205557, 26.5826283, -9.7642736, 26.6232662, -36.4438210, 36.3469009
19: -17.4946976, 9.3683844, -17.4334736, 9.3474312, -26.8421288, 26.8018570
20: -25.4978294, 3.7779419, -25.4318466, 3.8042712, -28.9801331, 28.9173508
21: -22.1076832, 8.0977411, -22.0273418, 8.1405563, -30.2482395, 30.1250839
22: -10.5039139, 21.3689995, -10.4593983, 21.3468189, -29.8938141, 29.8522034
23: -10.4013996, 18.1828766, -10.3549109, 18.1794662, -28.3491821, 28.2805939
24: -6.5345597, 22.4121056, -6.4913559, 22.4099312, -28.3643188, 28.3185425
25: -9.5755787, 22.9072876, -9.5230579, 22.8745117, -32.3748016, 32.3416138
26: -23.2675915, 24.4641380, -23.2164612, 24.4794197, -47.1555481, 47.0681152
27: -15.9501400, 17.4399681, -15.9039011, 17.4618340, -33.4119720, 33.3438683
28: -14.0422478, 19.4212513, -13.9930286, 19.4216652, -31.6685944, 31.6245270
29: -10.9387922, 15.4949865, -10.9157047, 15.4762630, -23.2046738, 23.2071609
30: -30.4322701, -0.0572770, -30.3715363, -0.0437346, -27.9979324, 27.9172745
31: -18.2500629, 10.7870197, -18.1864262, 10.7739639, -29.0240269, 28.9734459
32: -51.2538986, -16.0714321, -51.2342453, -16.0529671, -27.9190521, 27.8851318
33: -69.1069260, -11.9586535, -69.0743637, -12.0027676, -49.9888306, 50.0246429
34: -63.1614799, -21.4051704, -63.1429138, -21.3862629, -29.7202377, 29.6935501
35: -42.8499413, -0.4584222, -42.8281326, -0.4896159, -34.4838715, 34.5477524
36: -42.2887115, 2.8499475, -42.2572098, 2.8203723, -36.4229126, 36.4597931
37: -75.2371445, -18.9205112, -75.2182541, -18.9725742, -41.9307861, 42.0061035
38: -52.4500046, 2.1249623, -52.3707504, 2.1018329, -47.4566650, 47.4273376
39: -72.2949371, -13.6206245, -72.2671890, -13.6726179, -54.5337830, 54.5576019
40: -76.4690399, -36.6782608, -76.4642334, -36.7041206, -29.0274429, 29.0000305
41: -52.0294991, -11.1448746, -52.0096474, -11.1313286, -29.2911224, 29.2757492
42: -47.8030319, -16.1569157, -47.7853775, -16.1413383, -24.5989075, 24.5589981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=198, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1424
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1712

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0290262, upper bound: 15.0094219
time: 18.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 15, lower bound: -15.0329532, upper bound: 15.0329531
time: 26.49 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 47.10 seconds
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9403291, upper bound: 15.0059371
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9440285, upper bound: 15.0291943
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9721178, upper bound: 15.0060594
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9758613, upper bound: 15.0292965
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9762057, upper bound: 15.0091787
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9799717, upper bound: 15.0326491
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 47.10
Output dim: 15, lower bound: -15.0079195, upper bound: 15.0092852
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -15.0116703, upper bound: 15.0327333
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9552397, upper bound: 15.0060213
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9587635, upper bound: 15.0292695
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9870172, upper bound: 15.0061430
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9905808, upper bound: 15.0293759
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 47.10
Output dim: 15, lower bound: -14.9973826, upper bound: 15.0093171
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -15.0013553, upper bound: 15.0328687
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -15.0290262, upper bound: 14.9850822
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -15.0329532, upper bound: 15.0085055
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -15.0290262, upper bound: 15.0094219
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 47.10
Output dim: 15, lower bound: -15.0329532, upper bound: 15.0329531

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.7512569, 27.0347958, -5.7369947, 27.0519810, -32.8032379, 32.7717896
1: 0.0165944, 22.8734856, 0.0666909, 22.8669834, -20.6405487, 20.5765610
2: -0.8516312, 25.0493488, -0.8426330, 25.0545349, -23.5746384, 23.5454102
3: -12.6106634, 15.9146099, -12.5747375, 15.9241714, -23.2767487, 23.2399445
4: -6.2527065, 20.8066158, -6.2166853, 20.7860012, -22.9874420, 22.9557648
5: -11.8930969, 20.8859253, -11.8616352, 20.8867302, -30.6125717, 30.5555649
6: -66.1839600, -30.7731018, -66.1856995, -30.7525253, -24.6386185, 24.7396088
7: -17.8902969, 14.9079990, -17.8361778, 14.8979712, -27.3976517, 27.3329544
8: -19.2716465, 10.2216530, -19.2393608, 10.2254639, -24.5604172, 24.4887199
9: -7.2111311, 22.8570480, -7.1488500, 22.8523502, -29.0995178, 28.9887466
10: -31.5996704, 12.2215843, -31.5826969, 12.2401009, -39.0692291, 38.8930511
11: -24.6602516, 0.1306949, -24.6455383, 0.1460271, -23.3265533, 23.3593063
12: -49.0560722, -8.5541134, -49.0400696, -8.5867491, -34.0128632, 34.0628967
13: -28.4735184, 19.0971260, -28.5085659, 19.1152153, -47.5887337, 47.6056900
14: -41.7303200, 8.8307858, -41.7634735, 8.8549271, -43.9739380, 43.7902145
15: -1.1438847, 28.0458031, -1.1549430, 28.0607910, -24.6006851, 24.4831009
16: -26.7001190, 10.5093937, -26.6315536, 10.4965467, -34.7326660, 34.5790405
17: -39.7515869, 4.8829708, -39.7747192, 4.8629131, -35.8215790, 35.9092865
18: -9.6583529, 26.5023842, -9.6724653, 26.5151653, -36.1735191, 36.1748505
19: -17.3623009, 9.2653866, -17.3783855, 9.2536469, -26.6159477, 26.6437721
20: -25.3370686, 3.6550679, -25.3687973, 3.6778691, -28.6954498, 28.7515869
21: -21.9237480, 7.9327707, -21.9536934, 7.9879823, -29.9117298, 29.8864632
22: -10.4133549, 21.3193111, -10.4024830, 21.2828941, -29.6813660, 29.7103119
23: -10.2887278, 18.1031780, -10.2841892, 18.0771427, -28.1209869, 28.1297684
24: -6.4054923, 22.3417168, -6.4258556, 22.3361893, -28.1472931, 28.1845016
25: -9.4477243, 22.8119450, -9.4577103, 22.7867870, -32.1061630, 32.1517792
26: -23.1295338, 24.3974743, -23.1185551, 24.3627186, -46.8320770, 46.8560486
27: -15.8584156, 17.3616390, -15.8673935, 17.3851051, -33.2435226, 33.2290344
28: -13.9188538, 19.3478870, -13.9146471, 19.3130646, -31.4191589, 31.4780502
29: -10.8932753, 15.4368382, -10.8698740, 15.4038010, -23.0704498, 23.0738754
30: -30.2748260, -0.1739583, -30.2900467, -0.1655061, -27.7192841, 27.7594452
31: -18.0967999, 10.6414185, -18.1270599, 10.6621027, -28.7589035, 28.7684784
32: -51.1868820, -16.1590996, -51.1860428, -16.1573563, -27.6868439, 27.7674179
33: -69.0168152, -12.0294523, -69.0345154, -12.0612411, -49.8179474, 49.9166565
34: -63.0680504, -21.4502640, -63.0675659, -21.4899464, -29.5014496, 29.5794144
35: -42.7848587, -0.5003526, -42.7809143, -0.5465658, -34.3130798, 34.4502716
36: -42.2230301, 2.7919457, -42.2172661, 2.7377567, -36.2024078, 36.3617859
37: -75.1610794, -18.9615612, -75.1635895, -19.0202465, -41.7784424, 41.9256287
38: -52.2779312, 2.0149779, -52.2806664, 1.9676485, -47.0947113, 47.2335358
39: -72.2069321, -13.6712265, -72.2277222, -13.7039604, -54.3648071, 54.4582672
40: -76.4049377, -36.7206268, -76.4201660, -36.7387238, -28.9337921, 28.9149933
41: -51.9832153, -11.2079248, -51.9757538, -11.2207851, -29.0370026, 29.1696320
42: -47.7621765, -16.2131634, -47.7502251, -16.2176189, -24.4779739, 24.4971390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8592217, upper bound: 15.0046219
time: 33.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8592217, upper bound: 14.9816816
time: 18.10 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.7546206, 27.0436783, -5.8613720, 27.0855904, -32.8402100, 32.9050522
1: 0.0146439, 22.8836193, -0.0049486, 22.9068184, -20.6772995, 20.6664581
2: -0.8522668, 25.0616341, -0.9268768, 25.1022758, -23.6234055, 23.6462402
3: -12.6115103, 15.9208879, -12.6291428, 15.9511595, -23.3064804, 23.3038559
4: -6.2543726, 20.8179741, -6.3174820, 20.8293419, -23.0347290, 23.0714798
5: -11.8940220, 20.8958855, -11.9307709, 20.9264164, -30.6531143, 30.6382065
6: -66.1885681, -30.7709484, -66.2130585, -30.6750145, -24.7326355, 24.7631874
7: -17.8916779, 14.9170742, -17.8996124, 14.9341803, -27.4366074, 27.4095764
8: -19.2721596, 10.2354317, -19.3337784, 10.2812748, -24.6162720, 24.5990677
9: -7.2136149, 22.8643627, -7.2420683, 22.8797531, -29.1284790, 29.0907974
10: -31.6024132, 12.2227039, -31.6323891, 12.2635736, -39.1022491, 38.9578171
11: -24.6779060, 0.1319344, -24.7146339, 0.2293572, -23.4253693, 23.4249802
12: -49.0651207, -8.5517387, -49.0778046, -8.4966831, -34.1125183, 34.1018906
13: -28.4762154, 19.1098461, -28.5851288, 19.1766071, -47.6528244, 47.6949768
14: -41.7325439, 8.8327589, -41.8223267, 8.8754616, -44.0019073, 43.8555756
15: -1.1468272, 28.0579014, -1.2533245, 28.1125259, -24.6474838, 24.5962296
16: -26.7043915, 10.5158844, -26.6982346, 10.5278234, -34.7697754, 34.6632156
17: -39.7562408, 4.8838940, -39.8257599, 4.8976464, -35.8762665, 35.9710693
18: -9.6802797, 26.5032597, -9.7527666, 26.5882416, -36.2685204, 36.2560272
19: -17.3731270, 9.2654963, -17.4257278, 9.2996511, -26.6727791, 26.6912231
20: -25.3495541, 3.6566448, -25.4226818, 3.7486196, -28.7841187, 28.8051682
21: -21.9381256, 7.9335461, -22.0171318, 8.0636692, -30.0017948, 29.9506779
22: -10.4237156, 21.3197918, -10.4491625, 21.3266315, -29.7616272, 29.7505341
23: -10.3035469, 18.1046886, -10.3454227, 18.1466789, -28.2179413, 28.1867447
24: -6.4184132, 22.3425331, -6.4802246, 22.3785934, -28.2092743, 28.2349854
25: -9.4588013, 22.8133049, -9.5110531, 22.8339615, -32.1969452, 32.1993790
26: -23.1507416, 24.3978519, -23.2008629, 24.4511585, -46.9490662, 46.9373169
27: -15.8621006, 17.3633194, -15.8932734, 17.4290447, -33.2911453, 33.2565918
28: -13.9354239, 19.3487034, -13.9827852, 19.3909664, -31.5151215, 31.5447083
29: -10.9023809, 15.4372253, -10.9100046, 15.4563942, -23.1376343, 23.1127167
30: -30.2938156, -0.1713707, -30.3628235, -0.0922830, -27.8145218, 27.8324356
31: -18.1043453, 10.6420403, -18.1745872, 10.7071877, -28.8115330, 28.8166275
32: -51.1957397, -16.1552811, -51.2257843, -16.0815773, -27.7743530, 27.8047104
33: -69.0193405, -12.0272846, -69.0609207, -12.0286846, -49.8549957, 49.9476929
34: -63.0844421, -21.4484406, -63.1325912, -21.4021568, -29.6098022, 29.6380234
35: -42.7935944, -0.4992836, -42.8201485, -0.5030789, -34.3633270, 34.4901733
36: -42.2313004, 2.7930357, -42.2506180, 2.7998958, -36.2737732, 36.3956757
37: -75.1715393, -18.9606495, -75.2074890, -18.9848671, -41.8259583, 41.9735947
38: -52.2953529, 2.0176258, -52.3543015, 2.0617952, -47.2075043, 47.3060455
39: -72.2097626, -13.6711254, -72.2536926, -13.6868715, -54.3937073, 54.4877625
40: -76.4088211, -36.7186241, -76.4468842, -36.7153435, -28.9612885, 28.9443436
41: -51.9894180, -11.2062931, -52.0042114, -11.1525860, -29.1131897, 29.1977768
42: -47.7698135, -16.2111435, -47.7811279, -16.1595497, -24.5566025, 24.5276337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8910795, upper bound: 15.0047340
time: 22.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8910795, upper bound: 14.9817957
time: 16.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.8320408, 27.0781403, -5.7683029, 27.0539818, -32.8860245, 32.8464432
1: -0.0293815, 22.9115734, 0.0493641, 22.8684521, -20.6871262, 20.6331100
2: -0.8997931, 25.0941772, -0.8624127, 25.0558357, -23.6228333, 23.6099701
3: -12.6639023, 15.9631729, -12.5963469, 15.9254189, -23.3277893, 23.3111267
4: -6.3133020, 20.8345203, -6.2392230, 20.7870026, -23.0474625, 23.0075989
5: -11.9330597, 20.9238701, -11.8768520, 20.8885822, -30.6565933, 30.6102142
6: -66.2321396, -30.7098560, -66.1872177, -30.7293129, -24.7120361, 24.8007469
7: -17.9295330, 14.9348869, -17.8510685, 14.8990011, -27.4397736, 27.3775253
8: -19.3426418, 10.2766571, -19.2682266, 10.2273331, -24.6297302, 24.5739365
9: -7.2550197, 22.8856850, -7.1629806, 22.8539848, -29.1480331, 29.0477524
10: -31.6259270, 12.2522392, -31.5875301, 12.2439594, -39.1051025, 38.9372101
11: -24.7074661, 0.1732304, -24.6481781, 0.1617591, -23.3943405, 23.4015541
12: -49.0746765, -8.4993362, -49.0417824, -8.5696659, -34.0509644, 34.1199112
13: -28.5091438, 19.1436577, -28.5209732, 19.1201115, -47.6292572, 47.6646309
14: -41.7836075, 8.8807716, -41.7810173, 8.8569803, -44.0348053, 43.8644867
15: -1.2108402, 28.1090946, -1.1795197, 28.0640392, -24.6673050, 24.5708351
16: -26.7259560, 10.5359955, -26.6363621, 10.4990997, -34.7718735, 34.6287613
17: -39.7919273, 4.9191265, -39.7882233, 4.8648930, -35.8592224, 35.9579468
18: -9.7229052, 26.5350933, -9.6747236, 26.5273228, -36.2502289, 36.2098160
19: -17.4323978, 9.3135853, -17.3805141, 9.2737465, -26.7061443, 26.6940994
20: -25.4181805, 3.7044070, -25.3719463, 3.6978166, -28.7958679, 28.8009872
21: -22.0110130, 7.9950747, -21.9571342, 8.0136318, -30.0246449, 29.9522095
22: -10.4615860, 21.3465767, -10.4048996, 21.2928371, -29.7443085, 29.7474670
23: -10.3462639, 18.1410084, -10.2866249, 18.0911865, -28.1946030, 28.1687012
24: -6.4678750, 22.3721657, -6.4288464, 22.3479233, -28.2273865, 28.2193832
25: -9.5175152, 22.8556461, -9.4614038, 22.8028793, -32.1959000, 32.2029419
26: -23.1965370, 24.4322681, -23.1219330, 24.3755550, -46.9195862, 46.9058533
27: -15.9066296, 17.4023590, -15.8702412, 17.4004917, -33.3071213, 33.2725983
28: -13.9843779, 19.3899212, -13.9174376, 19.3294086, -31.5004501, 31.5216980
29: -10.9159565, 15.4654331, -10.8709431, 15.4120693, -23.1018524, 23.1015472
30: -30.3451271, -0.1292832, -30.2927570, -0.1494951, -27.8062134, 27.8058548
31: -18.1739616, 10.7018747, -18.1306553, 10.6865158, -28.8604774, 28.8325310
32: -51.2183723, -16.1179600, -51.1884232, -16.1456490, -27.7304535, 27.8088989
33: -69.0708466, -11.9974012, -69.0397110, -12.0504856, -49.8817596, 49.9521637
34: -63.1043854, -21.4334316, -63.0706215, -21.4844093, -29.5499191, 29.6031799
35: -42.8230896, -0.4805577, -42.7836151, -0.5403230, -34.3607025, 34.4761505
36: -42.2675095, 2.8292277, -42.2190018, 2.7511692, -36.2632294, 36.4007721
37: -75.2015457, -18.9412098, -75.1661606, -19.0149803, -41.8254395, 41.9471436
38: -52.3794060, 2.0767069, -52.2857742, 1.9895153, -47.2172852, 47.2962952
39: -72.2600174, -13.6511555, -72.2325745, -13.6997299, -54.4382629, 54.4890442
40: -76.4276733, -36.7062149, -76.4248581, -36.7350159, -28.9590607, 28.9333649
41: -52.0106125, -11.1734972, -51.9773293, -11.2099953, -29.0799561, 29.2040863
42: -47.7828789, -16.1854458, -47.7516594, -16.2096977, -24.5062027, 24.5244370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8962316, upper bound: 15.0111882
time: 21.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8962316, upper bound: 14.9849350
time: 25.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.8354082, 27.0869751, -5.8926868, 27.0875702, -32.9229774, 32.9796600
1: -0.0312924, 22.9216976, -0.0223141, 22.9082794, -20.7238922, 20.7230301
2: -0.9004476, 25.1064281, -0.9466743, 25.1035576, -23.6715393, 23.7108154
3: -12.6647053, 15.9694452, -12.6506891, 15.9524593, -23.3574982, 23.3750381
4: -6.3149490, 20.8458977, -6.3400369, 20.8303585, -23.0947571, 23.1233482
5: -11.9340019, 20.9338379, -11.9459991, 20.9282417, -30.6971283, 30.6929398
6: -66.2368164, -30.7076988, -66.2145920, -30.6518135, -24.8060684, 24.8243027
7: -17.9308357, 14.9440050, -17.9144878, 14.9352102, -27.4787369, 27.4541245
8: -19.3431549, 10.2904053, -19.3626251, 10.2831926, -24.6855927, 24.6843109
9: -7.2574477, 22.8930664, -7.2562242, 22.8814449, -29.1770248, 29.1498260
10: -31.6287022, 12.2532845, -31.6373558, 12.2674503, -39.1381226, 39.0020142
11: -24.7251434, 0.1744723, -24.7172489, 0.2450595, -23.4931564, 23.4672470
12: -49.0837669, -8.4969454, -49.0794067, -8.4794903, -34.1506500, 34.1589661
13: -28.5118217, 19.1564159, -28.5975685, 19.1815147, -47.6933365, 47.7539825
14: -41.7858276, 8.8827286, -41.8399277, 8.8775482, -44.0627747, 43.9298401
15: -1.2138124, 28.1211033, -1.2779160, 28.1156826, -24.7140656, 24.6839828
16: -26.7301712, 10.5425425, -26.7030029, 10.5303488, -34.8090363, 34.7129974
17: -39.7965698, 4.9200511, -39.8393173, 4.8996878, -35.9139099, 36.0196762
18: -9.7447891, 26.5359993, -9.7549906, 26.6003609, -36.3451500, 36.2909889
19: -17.4432850, 9.3136759, -17.4278603, 9.3197451, -26.7630310, 26.7415352
20: -25.4306850, 3.7059767, -25.4258137, 3.7685630, -28.8845215, 28.8545990
21: -22.0253334, 7.9957948, -22.0205822, 8.0892620, -30.1145954, 30.0163765
22: -10.4718351, 21.3469963, -10.4515667, 21.3365688, -29.8246460, 29.7876663
23: -10.3610601, 18.1425247, -10.3478765, 18.1607056, -28.2915497, 28.2256012
24: -6.4807696, 22.3729668, -6.4832354, 22.3903465, -28.2893600, 28.2698517
25: -9.5285454, 22.8570251, -9.5147457, 22.8501244, -32.2867508, 32.2505798
26: -23.2176781, 24.4326382, -23.2043419, 24.4640636, -47.0365295, 46.9870911
27: -15.9103165, 17.4040852, -15.8960791, 17.4444199, -33.3547363, 33.3001633
28: -14.0009747, 19.3907547, -13.9854984, 19.4072266, -31.5963593, 31.5883102
29: -10.9250889, 15.4658136, -10.9110813, 15.4646807, -23.1690216, 23.1404037
30: -30.3641281, -0.1267147, -30.3655167, -0.0762482, -27.9014816, 27.8788528
31: -18.1815319, 10.7025013, -18.1782303, 10.7315693, -28.9131012, 28.8807316
32: -51.2271919, -16.1141033, -51.2281723, -16.0698910, -27.8179703, 27.8461456
33: -69.0733948, -11.9952106, -69.0660629, -12.0179415, -49.9186859, 49.9831696
34: -63.1207428, -21.4317398, -63.1356125, -21.3966255, -29.6582489, 29.6618118
35: -42.8318024, -0.4794192, -42.8228531, -0.4968431, -34.4110413, 34.5160217
36: -42.2757530, 2.8303556, -42.2523079, 2.8132472, -36.3346558, 36.4346771
37: -75.2120667, -18.9402409, -75.2099991, -18.9794998, -41.8729553, 41.9951172
38: -52.3969421, 2.0792727, -52.3592949, 2.0836363, -47.3301544, 47.3687897
39: -72.2626190, -13.6509724, -72.2585297, -13.6826191, -54.4671478, 54.5184174
40: -76.4316177, -36.7042046, -76.4515686, -36.7115974, -28.9865417, 28.9627151
41: -52.0168724, -11.1718693, -52.0057449, -11.1417809, -29.1561661, 29.2322235
42: -47.7905579, -16.1834335, -47.7825813, -16.1516037, -24.5848694, 24.5549011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=143, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9280482, upper bound: 15.0112899
time: 29.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9280482, upper bound: 14.9850279
time: 20.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.8653088, 27.0742836, -5.7906132, 27.0558968, -32.9212036, 32.8648987
1: -0.0342779, 22.8920593, 0.0441282, 22.8701782, -20.6987152, 20.6332321
2: -0.9192965, 25.0869637, -0.8754430, 25.0574608, -23.6341629, 23.6174545
3: -12.6685944, 15.9415951, -12.6029663, 15.9276075, -23.3371201, 23.2481003
4: -6.3143597, 20.8150253, -6.2454376, 20.7879524, -23.0473022, 22.9972954
5: -11.9433889, 20.9112320, -11.8856831, 20.8910580, -30.6701050, 30.6009064
6: -66.2081909, -30.7124062, -66.1892700, -30.7239113, -24.7030945, 24.7947617
7: -17.9361839, 14.9218531, -17.8557186, 14.9007568, -27.4536285, 27.3554764
8: -19.3635101, 10.2689238, -19.2852325, 10.2306604, -24.6370544, 24.5461082
9: -7.2458639, 22.8760529, -7.1619234, 22.8604908, -29.1386719, 29.0467224
10: -31.6630230, 12.2792606, -31.5944633, 12.2659359, -39.1670151, 39.0697632
11: -24.7065525, 0.1848404, -24.6493015, 0.1717832, -23.3896103, 23.3837891
12: -49.0703735, -8.4924202, -49.0444679, -8.5616226, -34.0535049, 34.1295776
13: -28.5385666, 19.1709270, -28.5398903, 19.1246452, -47.6632118, 47.7108154
14: -41.7971954, 8.8724127, -41.7932358, 8.8590107, -44.0423889, 43.9254837
15: -1.2353191, 28.0971489, -1.1973877, 28.0660019, -24.6661453, 24.5666771
16: -26.7341156, 10.5215425, -26.6381302, 10.5020542, -34.7789307, 34.6414719
17: -39.8605309, 4.9533739, -39.8232155, 4.8673501, -35.9629974, 36.0346985
18: -9.7329531, 26.5488205, -9.6788731, 26.5378380, -36.2707901, 36.2276917
19: -17.4132538, 9.3199825, -17.3827534, 9.2811108, -26.6943645, 26.7027359
20: -25.4038754, 3.7268732, -25.3740520, 3.7134056, -28.7981873, 28.8085098
21: -22.0053005, 8.0344887, -21.9584999, 8.0390434, -30.0443439, 29.9929886
22: -10.4439325, 21.3412247, -10.4070301, 21.2930908, -29.7488861, 29.7503052
23: -10.3282547, 18.1433029, -10.2888680, 18.0956783, -28.1776047, 28.1610870
24: -6.4586143, 22.3807411, -6.4325509, 22.3556175, -28.2205353, 28.2320175
25: -9.4936705, 22.8619633, -9.4638805, 22.8108444, -32.1921997, 32.2176437
26: -23.1776714, 24.4285297, -23.1262188, 24.3780079, -46.9492645, 46.9131317
27: -15.8983479, 17.3972397, -15.8749123, 17.4023037, -33.3006516, 33.2721519
28: -13.9596624, 19.3780270, -13.9206409, 19.3273563, -31.4905930, 31.5051956
29: -10.9084291, 15.4667511, -10.8739376, 15.4151888, -23.1066742, 23.1084900
30: -30.3420105, -0.1047215, -30.2937698, -0.1333039, -27.8179626, 27.7936478
31: -18.1647034, 10.7257271, -18.1338310, 10.7043324, -28.8690357, 28.8595581
32: -51.2130585, -16.1167469, -51.1920013, -16.1407356, -27.7369537, 27.8058319
33: -69.0506668, -11.9935722, -69.0426559, -12.0463209, -49.8757172, 49.9572601
34: -63.1086884, -21.4238663, -63.0746765, -21.4798203, -29.5266495, 29.6107407
35: -42.8028717, -0.4797802, -42.7859077, -0.5394621, -34.3641663, 34.4812622
36: -42.2365036, 2.8110669, -42.2220764, 2.7447033, -36.2720642, 36.3863297
37: -75.1871796, -18.9411850, -75.1715546, -19.0138168, -41.8332672, 41.9500809
38: -52.3312302, 2.0604048, -52.2921677, 1.9855204, -47.1850739, 47.2915039
39: -72.2393799, -13.6412249, -72.2362518, -13.6941757, -54.4143066, 54.4960175
40: -76.4419861, -36.6959839, -76.4325714, -36.7334938, -28.9679947, 28.9487839
41: -51.9959106, -11.1812286, -51.9795990, -11.2106380, -29.1232147, 29.2118149
42: -47.7745552, -16.1869888, -47.7529678, -16.2076645, -24.4972992, 24.4957504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8758454, upper bound: 15.0064837
time: 27.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.8758454, upper bound: 14.9818310
time: 16.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.8686066, 27.0831375, -5.9150419, 27.0894947, -32.9580994, 32.9981804
1: -0.0362589, 22.9021835, -0.0276096, 22.9099808, -20.7354813, 20.7231598
2: -0.9199848, 25.0992393, -0.9597535, 25.1051750, -23.6829071, 23.7182999
3: -12.6694059, 15.9478855, -12.6573467, 15.9546576, -23.3667984, 23.3120804
4: -6.3160133, 20.8264484, -6.3462539, 20.8312874, -23.0945892, 23.1130219
5: -11.9443350, 20.9212036, -11.9548073, 20.9307766, -30.7105255, 30.6835785
6: -66.2128220, -30.7102470, -66.2166138, -30.6463604, -24.7971268, 24.8183365
7: -17.9375420, 14.9309196, -17.9191780, 14.9369459, -27.4925308, 27.4320908
8: -19.3640900, 10.2827158, -19.3796425, 10.2864647, -24.6928864, 24.6564255
9: -7.2483249, 22.8834114, -7.2551270, 22.8879700, -29.1676788, 29.1488342
10: -31.6657867, 12.2804060, -31.6442661, 12.2894306, -39.2001266, 39.1346283
11: -24.7242260, 0.1861138, -24.7183952, 0.2551377, -23.4884720, 23.4495201
12: -49.0794067, -8.4899874, -49.0821457, -8.4715195, -34.1533051, 34.1685944
13: -28.5413170, 19.1837234, -28.6165848, 19.1859894, -47.7273064, 47.8003082
14: -41.7994766, 8.8744946, -41.8522835, 8.8795347, -44.0704193, 43.9908752
15: -1.2382174, 28.1092224, -1.2957964, 28.1176434, -24.7128830, 24.6798744
16: -26.7384109, 10.5280666, -26.7047901, 10.5333471, -34.8161774, 34.7256165
17: -39.8651009, 4.9542580, -39.8742294, 4.9021006, -36.0177307, 36.0964432
18: -9.7548132, 26.5496712, -9.7591572, 26.6109657, -36.3657799, 36.3088303
19: -17.4241638, 9.3200893, -17.4300880, 9.3271465, -26.7513103, 26.7501774
20: -25.4163322, 3.7284129, -25.4279480, 3.7840993, -28.8868790, 28.8620987
21: -22.0196037, 8.0352545, -22.0219250, 8.1147194, -30.1343231, 30.0571785
22: -10.4542046, 21.3416576, -10.4537621, 21.3368320, -29.8292236, 29.7905197
23: -10.3430510, 18.1448021, -10.3500900, 18.1652031, -28.2745819, 28.2180328
24: -6.4714947, 22.3815594, -6.4869194, 22.3980408, -28.2824936, 28.2824707
25: -9.5047150, 22.8633575, -9.5171881, 22.8581066, -32.2830048, 32.2652588
26: -23.1988430, 24.4289627, -23.2086353, 24.4664783, -47.0662231, 46.9943695
27: -15.9020824, 17.3989201, -15.9007502, 17.4462261, -33.3483086, 33.2996712
28: -13.9762392, 19.3788624, -13.9887629, 19.4052334, -31.5864944, 31.5718460
29: -10.9175606, 15.4671783, -10.9141464, 15.4677534, -23.1738815, 23.1473465
30: -30.3609276, -0.1022282, -30.3665466, -0.0599885, -27.9132004, 27.8666229
31: -18.1722851, 10.7263107, -18.1813774, 10.7494278, -28.9217129, 28.9076881
32: -51.2218666, -16.1128902, -51.2317505, -16.0650139, -27.8244781, 27.8431091
33: -69.0531845, -11.9914045, -69.0689697, -12.0137100, -49.9126434, 49.9881897
34: -63.1250992, -21.4221668, -63.1396790, -21.3919773, -29.6350403, 29.6693726
35: -42.8116302, -0.4786959, -42.8251495, -0.4958909, -34.4144592, 34.5211563
36: -42.2447281, 2.8122263, -42.2553062, 2.8068283, -36.3435059, 36.4201813
37: -75.1976242, -18.9402103, -75.2154160, -18.9783955, -41.8807373, 41.9980469
38: -52.3486214, 2.0630674, -52.3656502, 2.0796990, -47.2978973, 47.3639832
39: -72.2420654, -13.6410789, -72.2621765, -13.6769896, -54.4431458, 54.5255585
40: -76.4459000, -36.6939240, -76.4592896, -36.7100830, -28.9954834, 28.9781799
41: -52.0021629, -11.1796036, -52.0079575, -11.1423607, -29.1993866, 29.2399597
42: -47.7822113, -16.1849670, -47.7838821, -16.1495209, -24.5759811, 24.5262108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9079472, upper bound: 15.0066218
time: 26.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9079472, upper bound: 14.9819468
time: 18.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.9463596, 27.1175308, -5.8220873, 27.0579109, -33.0042725, 32.9396172
1: -0.0804148, 22.9300613, 0.0266907, 22.8716431, -20.7454300, 20.6898193
2: -0.9676948, 25.1317406, -0.8953507, 25.0587864, -23.6827316, 23.6820450
3: -12.7220364, 15.9900885, -12.6246395, 15.9288626, -23.3883362, 23.3193359
4: -6.3757205, 20.8429756, -6.2682981, 20.7889652, -23.1076431, 23.0492744
5: -11.9836922, 20.9493103, -11.9009800, 20.8929043, -30.7145081, 30.6555634
6: -66.2561340, -30.6489677, -66.1908417, -30.7006931, -24.7766495, 24.8562508
7: -17.9755669, 14.9487705, -17.8705769, 14.9017544, -27.4960938, 27.4001083
8: -19.4358578, 10.3243389, -19.3147087, 10.2326107, -24.7067032, 24.6314621
9: -7.2889857, 22.9020348, -7.1760826, 22.8621960, -29.1874695, 29.1058502
10: -31.6856384, 12.3076296, -31.5993252, 12.2698460, -39.1986084, 39.1122055
11: -24.7537708, 0.2275219, -24.6519642, 0.1875744, -23.4575653, 23.4262848
12: -49.0893517, -8.4366474, -49.0461426, -8.5443001, -34.0932693, 34.1878357
13: -28.5748215, 19.2168217, -28.5524273, 19.1294823, -47.7043037, 47.7692490
14: -41.8509445, 8.9221992, -41.8110008, 8.8610573, -44.1044617, 44.0007401
15: -1.3024673, 28.1602211, -1.2220683, 28.0692215, -24.7332001, 24.6543922
16: -26.7602882, 10.5476818, -26.6429367, 10.5045900, -34.8180542, 34.6918945
17: -39.9013748, 4.9898243, -39.8368492, 4.8693261, -36.0023804, 36.0858231
18: -9.7974319, 26.5816822, -9.6811028, 26.5500393, -36.3474731, 36.2627869
19: -17.4832172, 9.3683147, -17.3848763, 9.3013382, -26.7845554, 26.7531910
20: -25.4849987, 3.7762995, -25.3772125, 3.7333500, -28.8985596, 28.8581772
21: -22.0925446, 8.0969486, -21.9619484, 8.0646858, -30.1572304, 30.0588970
22: -10.4921360, 21.3685799, -10.4094648, 21.3030758, -29.8118668, 29.7875900
23: -10.3856554, 18.1813354, -10.2913046, 18.1097279, -28.2511520, 28.2001572
24: -6.5209789, 22.4112511, -6.4355659, 22.3674393, -28.3007507, 28.2669678
25: -9.5634394, 22.9058228, -9.4675827, 22.8270969, -32.2821045, 32.2690277
26: -23.2445526, 24.4637413, -23.1295967, 24.3909569, -47.0366974, 46.9633636
27: -15.9463024, 17.4382668, -15.8777218, 17.4177284, -33.3640289, 33.3159866
28: -14.0249977, 19.4203949, -13.9233932, 19.3436985, -31.5716782, 31.5492020
29: -10.9293957, 15.4945297, -10.8750496, 15.4235668, -23.1373138, 23.1365433
30: -30.4123497, -0.0598733, -30.2964630, -0.1171808, -27.9050522, 27.8402634
31: -18.2418900, 10.7863703, -18.1374779, 10.7287493, -28.9706383, 28.9238472
32: -51.2449188, -16.0754051, -51.1943817, -16.1290627, -27.7811813, 27.8476334
33: -69.1042786, -11.9609165, -69.0477982, -12.0354595, -49.9391174, 49.9933929
34: -63.1449394, -21.4069290, -63.0777512, -21.4741516, -29.5750046, 29.6346207
35: -42.8411331, -0.4595556, -42.7886353, -0.5332189, -34.4115906, 34.5076218
36: -42.2803650, 2.8488002, -42.2237473, 2.7581792, -36.3326569, 36.4257050
37: -75.2266006, -18.9217186, -75.1741867, -19.0084476, -41.8806458, 41.9711456
38: -52.4326172, 2.1222367, -52.2971077, 2.0075469, -47.3077240, 47.3545685
39: -72.2920761, -13.6207724, -72.2410278, -13.6898422, -54.4890137, 54.5278931
40: -76.4650726, -36.6813698, -76.4372635, -36.7297554, -28.9934921, 28.9673843
41: -52.0231934, -11.1466351, -51.9810867, -11.1998405, -29.1643982, 29.2463608
42: -47.7953339, -16.1589832, -47.7544136, -16.1997166, -24.5257797, 24.5236816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=30, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9242008, upper bound: 15.0134654
time: 8.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9242008, upper bound: 14.9852456
time: 24.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.7914019, 27.0927811, -5.8622313, 27.0624619, -32.8538628, 32.9550133
1: 0.0326052, 22.8941765, -0.0207639, 22.8851376, -20.6367874, 20.7169495
2: -0.8858783, 25.1014175, -0.9602973, 25.0798283, -23.6197433, 23.7312317
3: -12.6146402, 15.9445076, -12.6545439, 15.9239302, -23.2748795, 23.3368225
4: -6.2623725, 20.7946434, -6.3417015, 20.7835350, -22.9935989, 23.0852051
5: -11.8901463, 20.9161606, -11.9433794, 20.9074059, -30.6022949, 30.6835327
6: -66.2411804, -30.7357559, -66.1793976, -30.7098789, -24.8380127, 24.7483406
7: -17.8610039, 14.9079876, -17.9031582, 14.9069080, -27.3666840, 27.4259644
8: -19.2778301, 10.2811890, -19.3376465, 10.2418013, -24.5518188, 24.6458702
9: -7.1665120, 22.8577366, -7.2238541, 22.8624306, -29.0476074, 29.1304855
10: -31.5925121, 12.2592611, -31.6092968, 12.2575731, -39.1002884, 39.0982819
11: -24.7252350, 0.1839805, -24.6953316, 0.2527192, -23.4950562, 23.3882866
12: -49.0489044, -8.5573940, -49.0431061, -8.5147705, -34.1024170, 34.0547028
13: -28.5506592, 19.1634140, -28.5739326, 19.1079235, -47.6585846, 47.7373466
14: -41.7547684, 8.9077053, -41.7344971, 8.8280296, -43.9887085, 44.0013962
15: -1.1930199, 28.1332932, -1.2381239, 28.0816975, -24.6234436, 24.7229195
16: -26.6563396, 10.4925947, -26.6819496, 10.5036163, -34.6525116, 34.6896210
17: -39.7652359, 4.9001484, -39.7800865, 4.8734674, -35.8951645, 35.9663010
18: -9.7965345, 26.5442429, -9.7388887, 26.6043205, -36.4008560, 36.2831306
19: -17.4313717, 9.2956991, -17.3901463, 9.3338108, -26.7651825, 26.6858444
20: -25.4647923, 3.7336936, -25.4129314, 3.7979360, -28.9273529, 28.8153534
21: -22.0672455, 8.0567856, -22.0020447, 8.1287947, -30.1960411, 30.0588303
22: -10.4272947, 21.3021965, -10.3967991, 21.3308849, -29.8107071, 29.7232666
23: -10.3239899, 18.1091690, -10.2789736, 18.1454544, -28.2413559, 28.1300430
24: -6.4989281, 22.3667126, -6.4659314, 22.3983002, -28.3202972, 28.2404556
25: -9.5002403, 22.8286819, -9.4572458, 22.8543510, -32.2907410, 32.1982269
26: -23.1927528, 24.3812256, -23.1463661, 24.4445934, -47.0498352, 46.9182129
27: -15.9095821, 17.4170361, -15.8527594, 17.4330177, -33.3425980, 33.2697945
28: -13.9631939, 19.3385105, -13.9142618, 19.3830528, -31.5605927, 31.4633636
29: -10.8549976, 15.4264908, -10.8428736, 15.4479084, -23.1007004, 23.0636139
30: -30.3942299, -0.1120455, -30.3445225, -0.0607085, -27.9381104, 27.8167953
31: -18.2015991, 10.7226200, -18.1516705, 10.7584820, -28.9600811, 28.8742905
32: -51.2329636, -16.1397400, -51.1950607, -16.1148605, -27.8369522, 27.7739792
33: -69.0568008, -12.0823889, -69.0128403, -12.1091013, -49.8345184, 49.8237762
34: -63.1248856, -21.4930992, -63.0993118, -21.4370403, -29.6380615, 29.5426331
35: -42.8071365, -0.5698223, -42.7754478, -0.5764532, -34.3636475, 34.3635559
36: -42.2384567, 2.7284951, -42.1951027, 2.7482042, -36.3151855, 36.2611694
37: -75.1813660, -19.0174599, -75.1828461, -18.9996414, -41.8591766, 41.8423767
38: -52.3774147, 1.9635820, -52.2842331, 1.9834113, -47.2743988, 47.1668854
39: -72.2508011, -13.7472973, -72.1867676, -13.8083115, -54.3563843, 54.3410492
40: -76.4554443, -36.7371902, -76.4521713, -36.7504807, -28.9361725, 28.9425735
41: -52.0067863, -11.2199993, -51.9706726, -11.1899185, -29.2054443, 29.1685791
42: -47.7775421, -16.1961365, -47.7710686, -16.1594124, -24.5472717, 24.4882851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9647958
time: 19.84 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9359578
time: 21.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.8321333, 27.0937786, -5.9452991, 27.0807877, -32.9129219, 33.0390778
1: 0.0205631, 22.8998432, -0.0445316, 22.8967628, -20.6603775, 20.7468948
2: -0.8949356, 25.1085758, -0.9790170, 25.0941582, -23.6451111, 23.7538223
3: -12.6254368, 15.9517231, -12.6785154, 15.9388466, -23.3028946, 23.3605194
4: -6.2743688, 20.8112297, -6.3683834, 20.8163223, -23.0326004, 23.1303177
5: -11.9028692, 20.9196033, -11.9696302, 20.9185295, -30.6579895, 30.7092056
6: -66.2438889, -30.6927948, -66.2145233, -30.6243191, -24.8533096, 24.8234558
7: -17.8751068, 14.9150162, -17.9335480, 14.9214478, -27.4079056, 27.4484558
8: -19.3133869, 10.2874680, -19.4084358, 10.2709846, -24.6174850, 24.7082672
9: -7.1878071, 22.8586693, -7.2682428, 22.8711586, -29.0792389, 29.1608582
10: -31.6109619, 12.2677212, -31.6478176, 12.2817631, -39.1398773, 39.1401978
11: -24.7351055, 0.1924803, -24.7191086, 0.2701850, -23.5205078, 23.4213333
12: -49.0535583, -8.5284424, -49.0690460, -8.4551964, -34.1618118, 34.1110153
13: -28.5617599, 19.2033844, -28.6292000, 19.1901207, -47.7518806, 47.8325844
14: -41.8194733, 8.9097881, -41.8673134, 8.8796148, -44.0995178, 44.0493240
15: -1.2339206, 28.1349373, -1.3191843, 28.1101360, -24.6945724, 24.7345963
16: -26.6688309, 10.4990492, -26.7091465, 10.5175085, -34.7297974, 34.7181473
17: -39.8071976, 4.9030561, -39.8643723, 4.9040165, -35.9682693, 36.0212402
18: -9.8075056, 26.5515499, -9.7612371, 26.6223831, -36.4298897, 36.3127861
19: -17.4426308, 9.2995148, -17.4158401, 9.3472128, -26.7898445, 26.7153549
20: -25.4699287, 3.7361071, -25.4256001, 3.8038926, -28.9597168, 28.8310242
21: -22.0773373, 8.0620375, -22.0240002, 8.1399002, -30.2172375, 30.0860367
22: -10.4490490, 21.3043385, -10.4405727, 21.3465824, -29.8463593, 29.7456665
23: -10.3526745, 18.1147938, -10.3374996, 18.1787167, -28.3040161, 28.1741333
24: -6.5070295, 22.3694153, -6.4828777, 22.4094334, -28.3326569, 28.2585068
25: -9.5249004, 22.8312340, -9.5061502, 22.8738556, -32.3250427, 32.2199020
26: -23.2178497, 24.3910980, -23.1970177, 24.4790974, -47.1074982, 46.9554901
27: -15.9335670, 17.4230499, -15.9026308, 17.4611702, -33.3947372, 33.3256798
28: -13.9919634, 19.3452473, -13.9752665, 19.4212475, -31.6276398, 31.5230865
29: -10.8844357, 15.4292536, -10.9002895, 15.4759617, -23.1605835, 23.0953598
30: -30.4040623, -0.1057601, -30.3650169, -0.0445991, -27.9758835, 27.8449402
31: -18.2127514, 10.7280035, -18.1773376, 10.7731152, -28.9858665, 28.9053421
32: -51.2384796, -16.1095104, -51.2314072, -16.0540237, -27.8538818, 27.8413925
33: -69.0623627, -12.0308981, -69.0627365, -12.0040979, -49.9318695, 49.9255219
34: -63.1307106, -21.4668503, -63.1336594, -21.3864975, -29.6621780, 29.6089172
35: -42.8118248, -0.5263474, -42.8157043, -0.4898903, -34.4327545, 34.4468155
36: -42.2455139, 2.7637494, -42.2415314, 2.8196559, -36.3747253, 36.3420410
37: -75.1851654, -19.0056362, -75.1997910, -18.9740009, -41.8945465, 41.9009018
38: -52.3856430, 2.0230560, -52.3488579, 2.1011877, -47.3650360, 47.2919464
39: -72.2577438, -13.6803722, -72.2560577, -13.6732292, -54.4841156, 54.4795074
40: -76.4592361, -36.7160454, -76.4634781, -36.7077827, -28.9749527, 28.9727631
41: -52.0106506, -11.1914244, -52.0042267, -11.1321936, -29.2126389, 29.2287369
42: -47.7811699, -16.1881065, -47.7806473, -16.1418648, -24.5843506, 24.5090675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=30, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9562656, upper bound: 14.9893924
time: 8.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9562656, upper bound: 14.9608561
time: 29.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.9089479, 27.1253490, -5.8633890, 27.0731354, -32.9820824, 32.9887390
1: -0.0702934, 22.9345779, -0.0212119, 22.8997993, -20.7586060, 20.7497559
2: -0.9593446, 25.1368961, -0.9608748, 25.0921249, -23.7061462, 23.7602768
3: -12.7120285, 15.9891644, -12.6550245, 15.9410429, -23.3899612, 23.3596115
4: -6.3653631, 20.8377590, -6.3424411, 20.7995377, -23.1159515, 23.1199341
5: -11.9718380, 20.9559174, -11.9438734, 20.9214611, -30.6993561, 30.7125626
6: -66.2580490, -30.6896725, -66.1830444, -30.7086201, -24.8554153, 24.8046494
7: -17.9628086, 14.9508219, -17.9036102, 14.9235086, -27.4938354, 27.4542313
8: -19.4008522, 10.3318787, -19.3383713, 10.2592239, -24.6968384, 24.6793938
9: -7.2701783, 22.9084702, -7.2249703, 22.8809547, -29.1848755, 29.1775436
10: -31.6699867, 12.3003120, -31.6106911, 12.2691612, -39.1920013, 39.1351776
11: -24.7615986, 0.2202752, -24.6973190, 0.2533510, -23.5309219, 23.4589386
12: -49.0937653, -8.4632664, -49.0578461, -8.5137138, -34.1336975, 34.1704636
13: -28.5664062, 19.1896286, -28.5737190, 19.1086769, -47.6750832, 47.7633476
14: -41.7885208, 8.9220629, -41.7372131, 8.8300362, -44.0216980, 44.0182343
15: -1.2644243, 28.1706543, -1.2394190, 28.0924110, -24.7088165, 24.7559395
16: -26.7520638, 10.5477867, -26.6824455, 10.5219736, -34.7779083, 34.7475510
17: -39.8640327, 4.9878278, -39.8036156, 4.8735499, -35.9840240, 36.0926208
18: -9.8083591, 26.5752335, -9.7391109, 26.6050453, -36.4134064, 36.3143463
19: -17.4828396, 9.3645544, -17.4065361, 9.3339758, -26.8168144, 26.7710915
20: -25.4923668, 3.7754772, -25.4183388, 3.7981987, -28.9549789, 28.8961105
21: -22.0968018, 8.0924826, -22.0033875, 8.1293030, -30.2261047, 30.0958710
22: -10.4806604, 21.3668327, -10.4123783, 21.3310394, -29.8565674, 29.8053665
23: -10.3717442, 18.1772118, -10.2940273, 18.1460705, -28.2854919, 28.2129745
24: -6.5258327, 22.4093590, -6.4729800, 22.3986645, -28.3503189, 28.2993622
25: -9.5498533, 22.9046345, -9.4719725, 22.8549061, -32.3385315, 32.2949905
26: -23.2406921, 24.4540615, -23.1612206, 24.4448910, -47.0960693, 47.0073547
27: -15.9259825, 17.4339294, -15.8537407, 17.4334679, -33.3594513, 33.2876701
28: -14.0127850, 19.4144783, -13.9305096, 19.3833885, -31.6006012, 31.5560303
29: -10.9091063, 15.4922237, -10.8578281, 15.4481554, -23.1445694, 23.1436615
30: -30.4214745, -0.0636282, -30.3488121, -0.0600188, -27.9625626, 27.8850708
31: -18.2383232, 10.7815533, -18.1593056, 10.7591791, -28.9975014, 28.9408588
32: -51.2482834, -16.1018639, -51.1977119, -16.1140900, -27.8518524, 27.8174286
33: -69.1013260, -12.0102749, -69.0242996, -12.1079741, -49.8787689, 49.9226837
34: -63.1555405, -21.4314175, -63.1082916, -21.4369450, -29.6592941, 29.6269836
35: -42.8451767, -0.5019534, -42.7876663, -0.5762675, -34.3928375, 34.4642639
36: -42.2816124, 2.8147073, -42.2106361, 2.7488518, -36.3445435, 36.3786774
37: -75.2332306, -18.9325676, -75.2009888, -18.9986553, -41.8928375, 41.9604797
38: -52.4417458, 2.0654063, -52.3059616, 1.9838028, -47.3298798, 47.3018799
39: -72.2878647, -13.6876125, -72.1977844, -13.8077011, -54.3901825, 54.4189148
40: -76.4651642, -36.7004967, -76.4527130, -36.7490463, -28.9821472, 28.9665604
41: -52.0255203, -11.1735287, -51.9759254, -11.1892405, -29.2335205, 29.2143021
42: -47.7993050, -16.1649761, -47.7757072, -16.1590729, -24.5673523, 24.5333481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9888340
time: 17.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9603210
time: 19.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9496918, 27.1263199, -5.9465218, 27.0914459, -33.0411377, 33.0728416
1: -0.0823088, 22.9402065, -0.0449913, 22.9114323, -20.7821655, 20.7797775
2: -0.9683018, 25.1440163, -0.9796281, 25.1065102, -23.7315063, 23.7829132
3: -12.7227936, 15.9963703, -12.6790018, 15.9559479, -23.4180069, 23.3832703
4: -6.3773112, 20.8543262, -6.3691645, 20.8322906, -23.1549149, 23.1650620
5: -11.9846144, 20.9593029, -11.9701424, 20.9325867, -30.7549973, 30.7382126
6: -66.2607422, -30.6468010, -66.2182312, -30.6230278, -24.8707047, 24.8798409
7: -17.9769268, 14.9578295, -17.9340267, 14.9379711, -27.5350494, 27.4767380
8: -19.4364281, 10.3381319, -19.4091415, 10.2884064, -24.7625427, 24.7417984
9: -7.2914486, 22.9094124, -7.2693353, 22.8896713, -29.2164307, 29.2079086
10: -31.6883736, 12.3087444, -31.6491718, 12.2933292, -39.2316589, 39.1770782
11: -24.7714691, 0.2287860, -24.7210732, 0.2709122, -23.5564041, 23.4919968
12: -49.0984039, -8.4343386, -49.0837555, -8.4541683, -34.1930237, 34.2268219
13: -28.5775795, 19.2295723, -28.6290474, 19.1908531, -47.7684326, 47.8586197
14: -41.8532295, 8.9241791, -41.8700371, 8.8815250, -44.1324463, 44.0662384
15: -1.3053946, 28.1722660, -1.3204474, 28.1208801, -24.7799377, 24.7675934
16: -26.7645340, 10.5542212, -26.7096443, 10.5358334, -34.8552551, 34.7760315
17: -39.9060020, 4.9907312, -39.8878975, 4.9041004, -36.0570221, 36.1475372
18: -9.8193598, 26.5825558, -9.7613859, 26.6231270, -36.4424858, 36.3439407
19: -17.4940720, 9.3684044, -17.4322052, 9.3473778, -26.8414497, 26.8006096
20: -25.4974937, 3.7778952, -25.4310932, 3.8041430, -28.9873505, 28.9117508
21: -22.1068687, 8.0976810, -22.0253448, 8.1404247, -30.2472935, 30.1230259
22: -10.5023508, 21.3689976, -10.4561262, 21.3467693, -29.8922272, 29.8277740
23: -10.4004374, 18.1828175, -10.3525600, 18.1792965, -28.3482056, 28.2570419
24: -6.5339174, 22.4120789, -6.4899302, 22.4098473, -28.3627014, 28.3174286
25: -9.5745201, 22.9072266, -9.5208874, 22.8743820, -32.3729095, 32.3165970
26: -23.2657547, 24.4640617, -23.2119408, 24.4794598, -47.1536255, 47.0446777
27: -15.9500294, 17.4399376, -15.9035339, 17.4616508, -33.4116821, 33.3434715
28: -14.0415812, 19.4212055, -13.9915009, 19.4215698, -31.6676483, 31.6158066
29: -10.9385490, 15.4949684, -10.9152231, 15.4762268, -23.2044907, 23.1753769
30: -30.4313164, -0.0573926, -30.3692379, -0.0439689, -28.0003128, 27.9132233
31: -18.2494507, 10.7869778, -18.1850166, 10.7738276, -29.0232773, 28.9719944
32: -51.2538223, -16.0715446, -51.2340469, -16.0532799, -27.8687668, 27.8848572
33: -69.1067963, -11.9587231, -69.0741577, -12.0028715, -49.9760437, 50.0243683
34: -63.1613426, -21.4051781, -63.1426926, -21.3863068, -29.6833725, 29.6932526
35: -42.8498344, -0.4584336, -42.8278542, -0.4896965, -34.4619598, 34.5475159
36: -42.2886353, 2.8499231, -42.2570572, 2.8202875, -36.4041138, 36.4595718
37: -75.2370987, -18.9207630, -75.2179565, -18.9729481, -41.9281921, 42.0189972
38: -52.4499588, 2.1247945, -52.3706093, 2.1016207, -47.4205475, 47.4270630
39: -72.2948685, -13.6206970, -72.2669754, -13.6726990, -54.5178833, 54.5573425
40: -76.4689484, -36.6793556, -76.4640198, -36.7063065, -29.0209503, 28.9967346
41: -52.0294113, -11.1449699, -52.0095139, -11.1315174, -29.2406693, 29.2745056
42: -47.8030014, -16.1569862, -47.7852974, -16.1415520, -24.6044540, 24.5541611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=197, inp2_unstable=197, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=145, inp2_unstable=145, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1424
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 857

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9562656, upper bound: 15.0135734
time: 20.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 15, lower bound: -14.9562656, upper bound: 14.9853355
time: 19.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 42.70 seconds
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8592217, upper bound: 15.0046219
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8592217, upper bound: 14.9816816
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8910795, upper bound: 15.0047340
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8910795, upper bound: 14.9817957
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8962316, upper bound: 15.0111882
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8962316, upper bound: 14.9849350
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9280482, upper bound: 15.0112899
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9280482, upper bound: 14.9850279
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8758454, upper bound: 15.0064837
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.8758454, upper bound: 14.9818310
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9079472, upper bound: 15.0066218
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9079472, upper bound: 14.9819468
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9242008, upper bound: 15.0134654
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9242008, upper bound: 14.9852456
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9647958
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9359578
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9562656, upper bound: 14.9893924
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9562656, upper bound: 14.9608561
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9888340
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9517247, upper bound: 14.9603210
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9562656, upper bound: 15.0135734
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 42.70
Output dim: 15, lower bound: -14.9562656, upper bound: 14.9853355

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 37.14 + 1972.82 = 2009.95 seconds
