## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 5)
Time budget: 3600 seconds
Split limit: 100
Threshold: 19.6865113824


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7322998, 24.7323036)
1: (-13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0198593, 16.0198631)
2: (-12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5284195, 15.5284157)
3: (-26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0969543, 20.0969582)
4: (-16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0408745, 21.0408783)
5: (-21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3915787, 19.3915787)
6: (-34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6224442, 22.6224403)
7: (-20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5227051, 21.5227051)
8: (-31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4536591, 26.4536591)
9: (-18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3851013, 23.3851013)
10: (-16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4907761, 25.4907761)
11: (-5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6712189, 17.6712227)
12: (-22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3685760, 29.3685684)
13: (-33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3678436, 30.3678360)
14: (-36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5356140, 43.5356140)
15: (-17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1119156, 24.1119194)
16: (-19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0168762, 21.0168800)
17: (-26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474)
18: (-7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2011566, 31.2011566)
19: (-1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5994358, 15.5994358)
20: (-7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5396423, 18.5396385)
21: (-5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0139999, 21.0139999)
22: (-2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6938057, 16.6938057)
23: (-4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8174934, 18.8174858)
24: (-2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7397270, 21.7397308)
25: (-5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0666885, 21.0666885)
26: (-7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5113754, 30.5113678)
27: (-5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6947441, 20.6947517)
28: (-2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6718521, 21.6718559)
29: (-2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6607780, 15.6607742)
30: (-9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2954025, 26.2954025)
31: (-5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6674004, 21.6673965)
32: (-28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0564461, 21.0564461)
33: (-50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8260040, 27.8260117)
34: (-45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4650192, 24.4650192)
35: (-32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2886353, 23.2886353)
36: (-29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4087448, 25.4087448)
37: (-46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1808167, 36.1808090)
38: (-40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8260040, 32.8260193)
39: (-50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5158768, 29.5158768)
40: (-48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0441437, 25.0441360)
41: (-28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7951050, 25.7951050)
42: (-32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9034767, 18.9034767)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.30 + 45.67 = 47.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 26, lower bound: -19.7062175, upper bound: 19.7062176

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1013
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1607

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6930864, upper bound: 19.6415027
time: 30.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7052723, upper bound: 19.7052719
time: 29.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 59.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 59.96
Output dim: 26, lower bound: -19.6930864, upper bound: 19.6415027
IS_A2, status: Status.UNKNOWN, split count: 1, time: 59.96
Output dim: 26, lower bound: -19.7052723, upper bound: 19.7052719

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -33.9180222, -0.6106336, -33.9249916, -0.6038275, -24.7144470, 24.7141266
1: -13.1139555, 7.4287882, -13.1217642, 7.4376283, -15.9996109, 15.9983521
2: -12.1595144, 8.6014595, -12.1659527, 8.6194286, -15.5107040, 15.4980736
3: -26.8507271, -2.2268167, -26.8765640, -2.2124290, -20.0418701, 20.0533371
4: -16.8615551, 11.0030289, -16.8671761, 11.0228376, -21.0129166, 20.9990921
5: -21.6327705, 3.2639780, -21.6358967, 3.2691560, -19.3796082, 19.3770981
6: -34.6313629, -7.5722547, -34.6774750, -7.5458679, -22.5326462, 22.5509186
7: -20.9211502, 6.2451644, -20.9286976, 6.2468443, -21.5040359, 21.5103836
8: -31.0351028, 4.9986925, -31.0386028, 5.0072942, -26.4372711, 26.4315033
9: -18.9395161, 8.0167599, -18.9680176, 8.0361309, -23.3257141, 23.3349152
10: -16.6475143, 10.9773464, -16.6811008, 11.0032711, -25.4181442, 25.4265785
11: -5.8967743, 16.3782177, -5.9123225, 16.3887997, -17.6338539, 17.6378403
12: -22.5476284, 13.6949425, -22.5920372, 13.7325401, -29.2785873, 29.2872467
13: -33.3705254, 6.7040086, -33.4281464, 6.7471237, -30.2536087, 30.2693100
14: -36.9562607, 8.4185867, -36.9871407, 8.4558716, -43.4552155, 43.4539337
15: -17.1779308, 9.4041214, -17.1870689, 9.4254227, -24.0835190, 24.0707245
16: -19.6463070, 3.7741442, -19.6681366, 3.7797201, -20.9759064, 20.9942245
17: -26.4135742, 7.7167521, -26.4496632, 7.7624645, -34.1760406, 34.1664162
18: -7.5678606, 25.3051643, -7.6165066, 25.3786564, -31.0790634, 31.0538635
19: -0.9659624, 16.0408859, -0.9965811, 16.0664368, -15.5359688, 15.5414734
20: -7.0380945, 12.3123474, -7.0538955, 12.3274527, -18.5036469, 18.5025711
21: -5.5083523, 16.2909355, -5.5308304, 16.3083172, -20.9675827, 20.9706573
22: -2.6287384, 16.8305264, -2.6543980, 16.8586159, -16.6359062, 16.6341629
23: -4.0007238, 17.6882343, -4.0222068, 17.7137070, -18.7650146, 18.7590294
24: -2.7515574, 22.1665039, -2.7911453, 22.2135410, -21.6525803, 21.6442757
25: -5.2707872, 18.2954483, -5.2935429, 18.3196239, -21.0160332, 21.0140572
26: -7.7806530, 24.4704552, -7.8260584, 24.5271721, -30.4041290, 30.3940353
27: -5.9175477, 18.0765018, -5.9610996, 18.1259651, -20.6017532, 20.5946999
28: -2.8858109, 20.4984436, -2.9171414, 20.5257645, -21.6090546, 21.6075668
29: -2.4388471, 17.1832848, -2.4657478, 17.2070961, -15.6074371, 15.6099052
30: -9.8087530, 18.6993637, -9.8259506, 18.7274017, -26.2432938, 26.2309494
31: -5.4645605, 17.6203651, -5.4962263, 17.6484089, -21.5999794, 21.6039429
32: -28.6231422, -1.3979864, -28.6890430, -1.3532138, -20.9247932, 20.9503250
33: -50.8211136, -11.7004490, -50.8681259, -11.6702061, -27.7415466, 27.7521210
34: -45.1782951, -13.7886963, -45.2284775, -13.7638121, -24.3842773, 24.4165802
35: -32.2865677, -2.9111443, -32.3235703, -2.8960264, -23.2347183, 23.2629318
36: -29.4366798, 2.3221846, -29.4670067, 2.3375092, -25.3507996, 25.3658218
37: -46.4162827, -5.4791589, -46.4415054, -5.4677172, -36.1326447, 36.1475525
38: -40.0944824, -2.7613778, -40.1249275, -2.7437749, -32.7621155, 32.7729645
39: -50.2945862, -7.9722905, -50.3499641, -7.9436460, -29.4014053, 29.4293137
40: -47.9760437, -17.6785831, -48.0125351, -17.6585846, -24.9798737, 25.0005798
41: -28.7192879, 0.9059501, -28.7675037, 0.9354339, -25.7022705, 25.7221146
42: -32.3922348, -9.5199738, -32.4494629, -9.4839382, -18.7905960, 18.8114777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=239, inp2_unstable=240, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1343

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6844025, upper bound: 19.6141200
time: 45.42 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6844025, upper bound: 19.6327675
time: 40.17 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -33.9296112, -0.5989256, -33.9304962, -0.5985758, -24.7308960, 24.7310410
1: -13.1279869, 7.4422493, -13.1286039, 7.4425812, -16.0157242, 16.0192337
2: -12.1677094, 8.6318254, -12.1679335, 8.6325741, -15.5275345, 15.5204544
3: -26.8971252, -2.2068653, -26.8992844, -2.2063103, -20.0610962, 20.0952721
4: -16.8720131, 11.0353708, -16.8724594, 11.0374908, -21.0392761, 21.0354652
5: -21.6381645, 3.2736430, -21.6386127, 3.2739277, -19.3903961, 19.3912048
6: -34.7147217, -7.5421648, -34.7165985, -7.5419579, -22.5812988, 22.6212883
7: -20.9334831, 6.2498240, -20.9348812, 6.2500830, -21.5187073, 21.5218353
8: -31.0413780, 5.0138135, -31.0414886, 5.0147038, -26.4530640, 26.4510574
9: -18.9921722, 8.0389681, -18.9928360, 8.0392714, -23.3550415, 23.3845291
10: -16.7112522, 11.0085220, -16.7120895, 11.0088415, -25.4573517, 25.4899292
11: -5.9219060, 16.3972397, -5.9224787, 16.3980064, -17.6710854, 17.6695747
12: -22.6316700, 13.7401180, -22.6327477, 13.7405558, -29.3168106, 29.3674698
13: -33.4778900, 6.7527781, -33.4795914, 6.7532854, -30.2882919, 30.3662491
14: -36.9979477, 8.4866219, -36.9989090, 8.4878483, -43.5409088, 43.5325775
15: -17.1907654, 9.4424314, -17.1912193, 9.4437723, -24.1104202, 24.1038628
16: -19.6841354, 3.7829401, -19.6858788, 3.7831440, -21.0171661, 21.0152283
17: -26.4578781, 7.8006516, -26.4582939, 7.8019676, -34.2598457, 34.2589455
18: -7.6227036, 25.4428120, -7.6233587, 25.4442310, -31.1994629, 31.1354446
19: -1.0052118, 16.0889645, -1.0058250, 16.0894756, -15.5983734, 15.5678787
20: -7.0593233, 12.3394127, -7.0597482, 12.3402119, -18.5386391, 18.5222054
21: -5.5412884, 16.3232040, -5.5419369, 16.3240967, -21.0128593, 21.0066071
22: -2.6595240, 16.8826485, -2.6598864, 16.8837929, -16.6929779, 16.6679688
23: -4.0280218, 17.7358208, -4.0284424, 17.7367477, -18.8164673, 18.7872162
24: -2.7970343, 22.2537155, -2.7975578, 22.2550049, -21.7383881, 21.6950073
25: -5.2999167, 18.3398590, -5.3003349, 18.3406887, -21.0656815, 21.0393257
26: -7.8330994, 24.5759354, -7.8335590, 24.5777168, -30.5100861, 30.4622192
27: -5.9673033, 18.1689110, -5.9677401, 18.1703644, -20.6933937, 20.6384773
28: -2.9230394, 20.5503979, -2.9233284, 20.5512066, -21.6708527, 21.6383858
29: -2.4707279, 17.2279797, -2.4711008, 17.2287693, -15.6598625, 15.6299248
30: -9.8337355, 18.7519703, -9.8341694, 18.7533321, -26.2942276, 26.2899475
31: -5.5051861, 17.6737118, -5.5060539, 17.6742725, -21.6659126, 21.6335335
32: -28.7451267, -1.3473225, -28.7466011, -1.3470302, -20.9916534, 21.0551300
33: -50.9084778, -11.6675224, -50.9099274, -11.6672363, -27.7475853, 27.8245544
34: -45.2698593, -13.7600994, -45.2713661, -13.7598400, -24.4362335, 24.4580307
35: -32.3525620, -2.8943436, -32.3541222, -2.8941381, -23.2994461, 23.2789154
36: -29.4891720, 2.3399549, -29.4904366, 2.3401351, -25.3677597, 25.4077530
37: -46.4629822, -5.4651861, -46.4643860, -5.4650183, -36.1669617, 36.1754608
38: -40.1492767, -2.7379198, -40.1509247, -2.7376289, -32.7801743, 32.8245773
39: -50.3976822, -7.9414291, -50.3994904, -7.9413090, -29.4222641, 29.5142136
40: -48.0440063, -17.6563473, -48.0454140, -17.6561966, -25.0069809, 25.0430756
41: -28.8060799, 0.9405026, -28.8072624, 0.9407630, -25.7567902, 25.7941284
42: -32.4968681, -9.4789047, -32.4990158, -9.4784184, -18.8581123, 18.9019394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=239, inp2_unstable=240, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1343

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6968397, upper bound: 19.6781215
time: 37.67 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6968397, upper bound: 19.6968393
time: 53.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 92.91 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 92.91
Output dim: 26, lower bound: -19.6844025, upper bound: 19.6141200
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 92.91
Output dim: 26, lower bound: -19.6844025, upper bound: 19.6327675
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 92.91
Output dim: 26, lower bound: -19.6968397, upper bound: 19.6781215
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 92.91
Output dim: 26, lower bound: -19.6968397, upper bound: 19.6968393

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -33.9070740, -0.6022959, -33.9203873, -0.6000233, -24.7102585, 24.7187347
1: -13.1243458, 7.4400887, -13.1269941, 7.4416409, -16.0113297, 16.0156708
2: -12.1550226, 8.6292362, -12.1624069, 8.6314716, -15.5143661, 15.5127296
3: -26.8723545, -2.2132940, -26.8885136, -2.2090874, -20.0337982, 20.0790405
4: -16.8458881, 11.0313625, -16.8610611, 11.0357666, -21.0105057, 21.0200386
5: -21.6156921, 3.2680960, -21.6286736, 3.2715144, -19.3640289, 19.3751106
6: -34.7120132, -7.5564995, -34.7154312, -7.5482278, -22.5714035, 22.6047440
7: -20.9144669, 6.2445035, -20.9264450, 6.2477312, -21.4972763, 21.5087128
8: -31.0281925, 5.0113926, -31.0357571, 5.0136433, -26.4396973, 26.4431610
9: -18.9659920, 8.0329132, -18.9813175, 8.0366545, -23.3267822, 23.3690796
10: -16.6986256, 10.9994144, -16.7065830, 11.0049000, -25.4411316, 25.4759369
11: -5.9189391, 16.3817024, -5.9211712, 16.3912354, -17.6614532, 17.6526146
12: -22.6253281, 13.6942377, -22.6300087, 13.7205915, -29.2893066, 29.3168335
13: -33.4632721, 6.7462420, -33.4732437, 6.7504478, -30.2733231, 30.3540268
14: -36.9790344, 8.4571953, -36.9906464, 8.4748201, -43.5096741, 43.4948273
15: -17.1731339, 9.4395752, -17.1835480, 9.4425287, -24.0928726, 24.0934677
16: -19.6797638, 3.7789438, -19.6839600, 3.7813857, -21.0090027, 21.0082932
17: -26.4504166, 7.7471113, -26.4550171, 7.7786570, -34.2290726, 34.2021294
18: -7.6132145, 25.4213562, -7.6192112, 25.4349327, -31.1818085, 31.1110611
19: -0.9978123, 16.0880585, -1.0025864, 16.0890713, -15.5890198, 15.5623779
20: -7.0523615, 12.3360081, -7.0567026, 12.3387127, -18.5222931, 18.5103569
21: -5.5356684, 16.3197708, -5.5394573, 16.3226051, -20.9994354, 20.9964142
22: -2.6534405, 16.8802567, -2.6572161, 16.8827515, -16.6859322, 16.6625404
23: -4.0230703, 17.7338448, -4.0262575, 17.7358971, -18.8105202, 18.7824020
24: -2.7894850, 22.2460957, -2.7942476, 22.2516861, -21.7272568, 21.6840210
25: -5.2925611, 18.3357010, -5.2971253, 18.3388672, -21.0559692, 21.0304947
26: -7.8229427, 24.5543861, -7.8291073, 24.5683670, -30.4909515, 30.4360352
27: -5.9585371, 18.1568260, -5.9638982, 18.1650829, -20.6790237, 20.6224403
28: -2.9165602, 20.5457134, -2.9205217, 20.5491753, -21.6617355, 21.6310120
29: -2.4659300, 17.2181358, -2.4690213, 17.2244797, -15.6509972, 15.6178207
30: -9.8308887, 18.7388096, -9.8329248, 18.7475719, -26.2851562, 26.2758789
31: -5.4963369, 17.6720123, -5.5021944, 17.6735382, -21.6513748, 21.6242561
32: -28.7380371, -1.3549633, -28.7435379, -1.3503814, -20.9801445, 21.0446205
33: -50.8829460, -11.6734991, -50.8985367, -11.6698666, -27.7264137, 27.8101273
34: -45.2595291, -13.7644329, -45.2668686, -13.7617111, -24.4214706, 24.4466400
35: -32.3415680, -2.8970101, -32.3493652, -2.8953106, -23.2813873, 23.2644043
36: -29.4807453, 2.3341889, -29.4868126, 2.3376479, -25.3529739, 25.3934021
37: -46.4509468, -5.4767342, -46.4592133, -5.4700541, -36.1377258, 36.1396561
38: -40.1379013, -2.7519407, -40.1459961, -2.7438679, -32.7600861, 32.8020706
39: -50.3672104, -7.9443984, -50.3859596, -7.9425731, -29.3921356, 29.4972992
40: -48.0384445, -17.6593094, -48.0429649, -17.6574535, -24.9922867, 25.0278015
41: -28.8013878, 0.9366231, -28.8052368, 0.9390635, -25.7460022, 25.7832870
42: -32.4949799, -9.4864092, -32.4981804, -9.4817333, -18.8503494, 18.8897438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=240, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1510

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6793594, upper bound: 19.6696229
time: 39.90 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6883166, upper bound: 19.6696229
time: 43.08 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -33.9258881, -0.5649633, -33.9255753, -0.6001823, -24.7244492, 24.7572479
1: -13.1276855, 7.4625068, -13.1272430, 7.4417100, -16.0162849, 16.0374222
2: -12.1616631, 8.6623259, -12.1635571, 8.6311111, -15.5197372, 15.5470772
3: -26.8833733, -2.1621513, -26.8907738, -2.2092066, -20.0445404, 20.1323280
4: -16.8602295, 11.0700111, -16.8630066, 11.0357523, -21.0225143, 21.0619049
5: -21.6258030, 3.3045225, -21.6308823, 3.2712736, -19.3724213, 19.4151535
6: -34.7378044, -7.5486975, -34.7152557, -7.5488586, -22.5960617, 22.6153679
7: -20.9256458, 6.2827921, -20.9292488, 6.2480249, -21.5046158, 21.5461655
8: -31.0348053, 5.0380216, -31.0363731, 5.0133929, -26.4453506, 26.4635620
9: -18.9850502, 8.0733271, -18.9844589, 8.0370607, -23.3460922, 23.4159088
10: -16.7163582, 11.0390882, -16.7049255, 11.0054417, -25.4616394, 25.5145721
11: -5.9560051, 16.3895149, -5.9209657, 16.3924103, -17.7017975, 17.6609879
12: -22.6753349, 13.7177429, -22.6294765, 13.7247267, -29.3444672, 29.3385162
13: -33.4762535, 6.7787843, -33.4741554, 6.7503848, -30.2915955, 30.3813629
14: -37.0334053, 8.4838161, -36.9915771, 8.4821796, -43.5731506, 43.5211945
15: -17.1871147, 9.4777393, -17.1846008, 9.4421740, -24.1058807, 24.1299133
16: -19.7033997, 3.8082910, -19.6839371, 3.7817583, -21.0387573, 21.0358620
17: -26.5085602, 7.7734261, -26.4556065, 7.7823830, -34.2909431, 34.2290344
18: -7.6585255, 25.4334793, -7.6191912, 25.4353771, -31.2254028, 31.1233215
19: -1.0260448, 16.0896416, -1.0030413, 16.0891075, -15.6143112, 15.5714607
20: -7.0733442, 12.3450212, -7.0570898, 12.3390446, -18.5321617, 18.5349541
21: -5.5596638, 16.3239594, -5.5392418, 16.3230515, -21.0174713, 21.0135765
22: -2.6749797, 16.8881645, -2.6576190, 16.8827744, -16.7081223, 16.6706581
23: -4.0509953, 17.7349911, -4.0265970, 17.7358170, -18.8398209, 18.7845192
24: -2.8192902, 22.2527180, -2.7948971, 22.2522392, -21.7586365, 21.6907310
25: -5.3256159, 18.3398724, -5.2976279, 18.3389664, -21.0888710, 21.0367050
26: -7.8892879, 24.5683670, -7.8293295, 24.5703106, -30.5602493, 30.4494476
27: -5.9923224, 18.1638451, -5.9640288, 18.1656036, -20.7157822, 20.6306000
28: -2.9498739, 20.5489388, -2.9208751, 20.5495453, -21.6931915, 21.6358795
29: -2.5114508, 17.2259026, -2.4694080, 17.2265015, -15.6995697, 15.6241417
30: -9.8579063, 18.7463493, -9.8329620, 18.7486916, -26.3118134, 26.2859039
31: -5.5209370, 17.6802502, -5.5026283, 17.6737499, -21.6707077, 21.6436501
32: -28.7518654, -1.3260469, -28.7435513, -1.3495774, -20.9930801, 21.0769348
33: -50.9054756, -11.6246948, -50.9045982, -11.6695623, -27.7423744, 27.8611755
34: -45.2741737, -13.7361374, -45.2670059, -13.7614250, -24.4467010, 24.4601135
35: -32.3528404, -2.8854265, -32.3484993, -2.8955870, -23.3043251, 23.2654953
36: -29.5092793, 2.3378420, -29.4872780, 2.3370857, -25.3885880, 25.3953857
37: -46.4841042, -5.4703155, -46.4599762, -5.4708962, -36.2033234, 36.1422272
38: -40.1639175, -2.7436857, -40.1463089, -2.7438774, -32.7959137, 32.8098145
39: -50.3990211, -7.9033504, -50.3927956, -7.9425058, -29.4173813, 29.5405884
40: -48.0465088, -17.6551113, -48.0431290, -17.6574688, -25.0125885, 25.0272827
41: -28.8206520, 0.9429107, -28.8052521, 0.9387670, -25.7706299, 25.7875137
42: -32.5074463, -9.4771461, -32.4979744, -9.4825602, -18.8725662, 18.8989449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=240, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1510

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6793594, upper bound: 19.6883162
time: 29.07 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6883166, upper bound: 19.6883162
time: 34.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 65.61 seconds
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 65.61
Output dim: 26, lower bound: -19.6793594, upper bound: 19.6696229
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 65.61
Output dim: 26, lower bound: -19.6883166, upper bound: 19.6696229
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 65.61
Output dim: 26, lower bound: -19.6793594, upper bound: 19.6883162
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 65.61
Output dim: 26, lower bound: -19.6883166, upper bound: 19.6883162

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -33.9030952, -0.6027021, -33.9246178, -0.5878801, -24.7173767, 24.7208557
1: -13.1238260, 7.4395452, -13.1330557, 7.4589524, -16.0292549, 16.0196953
2: -12.1527290, 8.6286440, -12.1651707, 8.6519432, -15.5360260, 15.5153732
3: -26.8721581, -2.2145905, -26.8969250, -2.1749253, -20.0606003, 20.0924263
4: -16.8422432, 11.0307941, -16.8661556, 11.0581989, -21.0284271, 21.0223045
5: -21.6115265, 3.2673054, -21.6264820, 3.2944765, -19.3872147, 19.3726120
6: -34.7094154, -7.5569673, -34.7246895, -7.5041056, -22.6162262, 22.6050644
7: -20.9114399, 6.2438960, -20.9298630, 6.2651644, -21.5150681, 21.5134583
8: -31.0279694, 5.0102005, -31.0638084, 5.0307813, -26.4534607, 26.4669189
9: -18.9655476, 8.0300732, -19.0523834, 8.0416574, -23.3205185, 23.4389648
10: -16.6980820, 10.9952173, -16.8476448, 11.0070858, -25.4309082, 25.6167183
11: -5.9178405, 16.3791733, -5.9747486, 16.3914051, -17.6579361, 17.7018890
12: -22.6249199, 13.6898708, -22.6625061, 13.7203560, -29.2874527, 29.3501663
13: -33.4595299, 6.7450790, -33.4747353, 6.7984061, -30.3062515, 30.3504105
14: -36.9776764, 8.4546795, -37.0986404, 8.4788685, -43.5042877, 43.6009216
15: -17.1724110, 9.4383593, -17.2206535, 9.4704590, -24.1093750, 24.1410713
16: -19.6785965, 3.7779164, -19.7538090, 3.7880182, -21.0084381, 21.0753403
17: -26.4478035, 7.7458916, -26.4703407, 7.7936211, -34.2414246, 34.2162323
18: -7.6097488, 25.4209652, -7.6609421, 25.4760036, -31.2376862, 31.1336212
19: -0.9958096, 16.0879669, -1.0308571, 16.0908337, -15.5948486, 15.5775967
20: -7.0514507, 12.3344479, -7.0805383, 12.3401289, -18.5241165, 18.5268784
21: -5.5345325, 16.3187313, -5.5929518, 16.3256741, -20.9895096, 21.0291176
22: -2.6525006, 16.8799191, -2.6694846, 16.8962612, -16.6893730, 16.6867638
23: -4.0219393, 17.7319832, -4.0464649, 17.7345963, -18.8085976, 18.7951355
24: -2.7861519, 22.2459164, -2.8005934, 22.2543259, -21.7266045, 21.6882744
25: -5.2907891, 18.3341999, -5.3146982, 18.3444729, -21.0550461, 21.0483208
26: -7.8213787, 24.5523987, -7.9004869, 24.5780792, -30.4916992, 30.4994507
27: -5.9553828, 18.1565475, -5.9738145, 18.1761780, -20.6866608, 20.6279831
28: -2.9154577, 20.5439358, -2.9353161, 20.5522518, -21.6676025, 21.6296425
29: -2.4648600, 17.2167931, -2.4889822, 17.2278366, -15.6509018, 15.6425400
30: -9.8295727, 18.7359486, -9.8508511, 18.7505188, -26.2841187, 26.2912750
31: -5.4933796, 17.6718502, -5.5310812, 17.6769981, -21.6524124, 21.6527710
32: -28.7369022, -1.3560596, -28.7535877, -1.3219175, -21.0045776, 21.0418358
33: -50.8805351, -11.6739235, -50.8989182, -11.5964251, -27.7957573, 27.7847290
34: -45.2580757, -13.7649384, -45.2741547, -13.7058029, -24.4689255, 24.4401169
35: -32.3395767, -2.8971741, -32.3536797, -2.8293428, -23.3436584, 23.2614098
36: -29.4773960, 2.3340244, -29.4931908, 2.4148998, -25.4307098, 25.3818130
37: -46.4465370, -5.4770436, -46.4782600, -5.3917255, -36.2092819, 36.1487427
38: -40.1338234, -2.7522659, -40.1622734, -2.6419501, -32.8570557, 32.7970123
39: -50.3627052, -7.9447131, -50.3967094, -7.8726702, -29.4600601, 29.4793777
40: -48.0362091, -17.6596489, -48.0531845, -17.6176243, -25.0284271, 25.0258331
41: -28.7988186, 0.9361653, -28.8136368, 0.9898257, -25.7987289, 25.7864151
42: -32.4945068, -9.4885225, -32.5084686, -9.4706593, -18.8509293, 18.9102821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1510

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1607

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6153853, upper bound: 19.6574361
time: 34.88 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6244809, upper bound: 19.6574366
time: 36.09 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -33.9186974, -0.5691004, -33.9120102, -0.6080899, -24.7084351, 24.7381020
1: -13.1259727, 7.4587865, -13.1240578, 7.4345732, -16.0051498, 16.0288277
2: -12.1573172, 8.6584625, -12.1564989, 8.6243534, -15.5084038, 15.5356865
3: -26.8821945, -2.1682758, -26.8890648, -2.2206402, -20.0276413, 20.1219025
4: -16.8492680, 11.0653858, -16.8459873, 11.0272236, -21.0024719, 21.0387573
5: -21.6196041, 3.2987881, -21.6211395, 3.2606206, -19.3545303, 19.3978004
6: -34.7144966, -7.5515928, -34.6705513, -7.5541620, -22.5655518, 22.5656128
7: -20.9190884, 6.2786422, -20.9187737, 6.2404480, -21.4909058, 21.5319252
8: -31.0322647, 5.0248756, -31.0316963, 4.9873376, -26.4170227, 26.4448700
9: -18.9813824, 8.0444212, -18.9779778, 7.9799805, -23.2846298, 23.3803558
10: -16.7105064, 10.9903307, -16.6942310, 10.9157085, -25.3666916, 25.4559097
11: -5.9502754, 16.3789120, -5.9100900, 16.3749199, -17.6786537, 17.6396637
12: -22.6715736, 13.7057114, -22.6222801, 13.7069244, -29.3208694, 29.3179321
13: -33.4631729, 6.7699409, -33.4494324, 6.7343545, -30.2574921, 30.3446350
14: -37.0229416, 8.4386549, -36.9724731, 8.4003248, -43.4810791, 43.4575806
15: -17.1817665, 9.4604454, -17.1749287, 9.4086418, -24.0606155, 24.0987892
16: -19.6984596, 3.7935910, -19.6751003, 3.7527821, -21.0049744, 21.0109863
17: -26.4953194, 7.7606373, -26.4303360, 7.7612171, -34.2565384, 34.1909714
18: -7.6462078, 25.4302673, -7.5958710, 25.4296246, -31.1981659, 31.0798874
19: -1.0146732, 16.0885105, -0.9829502, 16.0868301, -15.5924492, 15.5379391
20: -7.0649233, 12.3419647, -7.0419035, 12.3333683, -18.5145721, 18.5116386
21: -5.5495377, 16.3107185, -5.5204601, 16.2968998, -20.9812851, 20.9780998
22: -2.6663723, 16.8852005, -2.6429920, 16.8772278, -16.6860504, 16.6477947
23: -4.0440111, 17.7287254, -4.0135970, 17.7250824, -18.8217850, 18.7621651
24: -2.8030643, 22.2515373, -2.7639389, 22.2502747, -21.7394485, 21.6586227
25: -5.3173647, 18.3328476, -5.2820134, 18.3264236, -21.0621796, 21.0100594
26: -7.8786221, 24.5525150, -7.8098297, 24.5420074, -30.5246124, 30.4142838
27: -5.9746919, 18.1625919, -5.9332285, 18.1632652, -20.6954651, 20.5979500
28: -2.9414988, 20.5446491, -2.9062982, 20.5412788, -21.6680145, 21.6043358
29: -2.5047035, 17.2153072, -2.4579320, 17.2072277, -15.6721115, 15.6015205
30: -9.8523769, 18.7386627, -9.8223658, 18.7344131, -26.2919083, 26.2678604
31: -5.5054846, 17.6792030, -5.4745598, 17.6717834, -21.6534729, 21.6153641
32: -28.7356243, -1.3286147, -28.7134628, -1.3544598, -20.9723129, 21.0456619
33: -50.8644676, -11.6282272, -50.8285294, -11.6755257, -27.6931877, 27.7800369
34: -45.2508316, -13.7396774, -45.2222443, -13.7678633, -24.4161606, 24.4123383
35: -32.3239594, -2.8879237, -32.2940483, -2.9001360, -23.2704544, 23.2086792
36: -29.4710598, 2.3356924, -29.4155064, 2.3330159, -25.3451614, 25.3194275
37: -46.4435921, -5.4731498, -46.3804703, -5.4758530, -36.1548309, 36.0567703
38: -40.1170197, -2.7462893, -40.0546379, -2.7486324, -32.7426453, 32.7140884
39: -50.3528595, -7.9058075, -50.3075562, -7.9467750, -29.3673859, 29.4554520
40: -48.0190659, -17.6571712, -47.9912109, -17.6611576, -24.9810486, 24.9731979
41: -28.7951927, 0.9402766, -28.7561054, 0.9338636, -25.7379837, 25.7332382
42: -32.5019531, -9.4837532, -32.4874458, -9.4931526, -18.8472824, 18.8755112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1510

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1607

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6153853, upper bound: 19.6760096
time: 34.14 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6153853, upper bound: 19.6883166
time: 34.05 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -33.9219437, -0.5653319, -33.9297981, -0.5879638, -24.7315903, 24.7593536
1: -13.1271696, 7.4619823, -13.1332893, 7.4590530, -16.0342026, 16.0414085
2: -12.1593342, 8.6617222, -12.1663208, 8.6515503, -15.5413933, 15.5497131
3: -26.8831177, -2.1633911, -26.8991852, -2.1750226, -20.0713043, 20.1457367
4: -16.8566036, 11.0694504, -16.8681087, 11.0582294, -21.0403976, 21.0642014
5: -21.6216393, 3.3037586, -21.6286926, 3.2942228, -19.3956146, 19.4126816
6: -34.7351913, -7.5491509, -34.7244720, -7.5047960, -22.6409378, 22.6156960
7: -20.9226418, 6.2821574, -20.9326134, 6.2653837, -21.5223465, 21.5508499
8: -31.0345440, 5.0369267, -31.0644264, 5.0305281, -26.4591064, 26.4873123
9: -18.9845886, 8.0704660, -19.0555534, 8.0420914, -23.3398285, 23.4858170
10: -16.7158356, 11.0348730, -16.8459301, 11.0076523, -25.4513855, 25.6553383
11: -5.9549055, 16.3869743, -5.9745259, 16.3925495, -17.6982956, 17.7102623
12: -22.6749573, 13.7133408, -22.6619797, 13.7245035, -29.3425598, 29.3718796
13: -33.4725380, 6.7776699, -33.4756470, 6.7983117, -30.3245316, 30.3777161
14: -37.0320702, 8.4813576, -37.0995255, 8.4862709, -43.5678253, 43.6274261
15: -17.1864033, 9.4764977, -17.2217140, 9.4700394, -24.1223907, 24.1774864
16: -19.7022343, 3.8072803, -19.7537613, 3.7883890, -21.0382080, 21.1028824
17: -26.5059814, 7.7722015, -26.4709301, 7.7973490, -34.3033295, 34.2431335
18: -7.6551275, 25.4331093, -7.6608639, 25.4764938, -31.2812653, 31.1459045
19: -1.0240688, 16.0895500, -1.0312948, 16.0908718, -15.6201324, 15.5866661
20: -7.0724273, 12.3435020, -7.0809140, 12.3404818, -18.5339737, 18.5514526
21: -5.5585423, 16.3229408, -5.5927391, 16.3261356, -21.0075684, 21.0462341
22: -2.6740389, 16.8878441, -2.6698470, 16.8962803, -16.7115631, 16.6948910
23: -4.0498638, 17.7331562, -4.0468302, 17.7345314, -18.8379097, 18.7972488
24: -2.8159361, 22.2525120, -2.8012524, 22.2548733, -21.7579460, 21.6949387
25: -5.3238807, 18.3383522, -5.3152075, 18.3445740, -21.0879707, 21.0545120
26: -7.8876781, 24.5663986, -7.9007177, 24.5800285, -30.5609665, 30.5128632
27: -5.9891558, 18.1635513, -5.9739351, 18.1766701, -20.7233810, 20.6361656
28: -2.9487777, 20.5471191, -2.9356327, 20.5526390, -21.6990662, 21.6344910
29: -2.5103498, 17.2245445, -2.4893794, 17.2298374, -15.6994286, 15.6488419
30: -9.8566217, 18.7434692, -9.8508854, 18.7516136, -26.3107758, 26.3012772
31: -5.5180035, 17.6801090, -5.5314980, 17.6772156, -21.6717834, 21.6720963
32: -28.7507286, -1.3271370, -28.7535801, -1.3211341, -21.0174866, 21.0741196
33: -50.9030533, -11.6251259, -50.9049835, -11.5961952, -27.8117218, 27.8357086
34: -45.2727203, -13.7366447, -45.2742920, -13.7054758, -24.4942169, 24.4536438
35: -32.3508072, -2.8856153, -32.3528595, -2.8296156, -23.3665810, 23.2624893
36: -29.5058708, 2.3376532, -29.4936619, 2.4143639, -25.4663315, 25.3837738
37: -46.4796562, -5.4706459, -46.4789963, -5.3925285, -36.2749405, 36.1513214
38: -40.1598206, -2.7440333, -40.1625519, -2.6419010, -32.8928680, 32.8047638
39: -50.3944931, -7.9036269, -50.4034996, -7.8726435, -29.4852905, 29.5225220
40: -48.0442581, -17.6554852, -48.0533257, -17.6176491, -25.0487518, 25.0253525
41: -28.8181076, 0.9424553, -28.8136330, 0.9895449, -25.8233490, 25.7906342
42: -32.5069809, -9.4792490, -32.5082817, -9.4714775, -18.8731384, 18.9194908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1510

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1607

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6244809, upper bound: 19.6760096
time: 35.82 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6244809, upper bound: 19.6760100
time: 35.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 73.37 seconds
IS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 73.37
Output dim: 26, lower bound: -19.6153853, upper bound: 19.6574361
IS_A2_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 73.37
Output dim: 26, lower bound: -19.6244809, upper bound: 19.6574366
IS_A2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 73.37
Output dim: 26, lower bound: -19.6153853, upper bound: 19.6760096
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 73.37
Output dim: 26, lower bound: -19.6153853, upper bound: 19.6883166
IS_A2_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 73.37
Output dim: 26, lower bound: -19.6244809, upper bound: 19.6760096
IS_A2_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 73.37
Output dim: 26, lower bound: -19.6244809, upper bound: 19.6760100

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -33.9186974, -0.5691004, -33.9111443, -0.6084394, -24.7077255, 24.7372437
1: -13.1259727, 7.4587865, -13.1234474, 7.4342246, -16.0048294, 16.0250206
2: -12.1573172, 8.6584625, -12.1562767, 8.6235943, -15.5010605, 15.5354424
3: -26.8821945, -2.1682758, -26.8868980, -2.2211943, -20.0270844, 20.0871811
4: -16.8492680, 11.0653858, -16.8455238, 11.0250959, -20.9981918, 21.0382996
5: -21.6196041, 3.2987881, -21.6207294, 3.2603340, -19.3547287, 19.3971786
6: -34.7144966, -7.5515928, -34.6686325, -7.5543499, -22.5653458, 22.5254021
7: -20.9190884, 6.2786422, -20.9173431, 6.2401533, -21.4906540, 21.5285339
8: -31.0322647, 5.0248756, -31.0315819, 4.9863815, -26.4148788, 26.4447098
9: -18.9813824, 8.0444212, -18.9772949, 7.9796805, -23.2843704, 23.3506470
10: -16.7105064, 10.9903307, -16.6933899, 10.9154091, -25.3663635, 25.4230423
11: -5.9502754, 16.3789120, -5.9095249, 16.3741417, -17.6779137, 17.6404114
12: -22.6715736, 13.7057114, -22.6211929, 13.7064886, -29.3204727, 29.2668304
13: -33.4631729, 6.7699409, -33.4477005, 6.7337914, -30.2569809, 30.2661972
14: -37.0229416, 8.4386549, -36.9715195, 8.3991261, -43.4797668, 43.4645691
15: -17.1817665, 9.4604454, -17.1744404, 9.4073076, -24.0535889, 24.0983200
16: -19.6984596, 3.7935910, -19.6733475, 3.7525816, -21.0046539, 21.0125275
17: -26.4953194, 7.7606373, -26.4298820, 7.7598839, -34.2552032, 34.1905212
18: -7.6462078, 25.4302673, -7.5951958, 25.4281960, -31.1335144, 31.0792999
19: -1.0146732, 16.0885105, -0.9823360, 16.0863190, -15.5613785, 15.5373573
20: -7.0649233, 12.3419647, -7.0414672, 12.3325682, -18.4976730, 18.5112381
21: -5.5495377, 16.3107185, -5.5198269, 16.2960033, -20.9744415, 20.9775276
22: -2.6663723, 16.8852005, -2.6426005, 16.8760834, -16.6606827, 16.6474419
23: -4.0440111, 17.7287254, -4.0132003, 17.7241707, -18.7921371, 18.7617645
24: -2.8030643, 22.2515373, -2.7634144, 22.2489815, -21.6955338, 21.6580963
25: -5.3173647, 18.3328476, -5.2815757, 18.3255901, -21.0353622, 21.0096054
26: -7.8786221, 24.5525150, -7.8093576, 24.5401993, -30.4763031, 30.4138107
27: -5.9746919, 18.1625919, -5.9327788, 18.1618042, -20.6401367, 20.5975494
28: -2.9414988, 20.5446491, -2.9059491, 20.5404644, -21.6351929, 21.6039734
29: -2.5047035, 17.2153072, -2.4575510, 17.2064381, -15.6417809, 15.6011353
30: -9.8523769, 18.7386627, -9.8219490, 18.7330189, -26.2872086, 26.2674637
31: -5.5054846, 17.6792030, -5.4736738, 17.6712093, -21.6202431, 21.6145020
32: -28.7356243, -1.3286147, -28.7119942, -1.3548021, -20.9720230, 20.9819527
33: -50.8644676, -11.6282272, -50.8271103, -11.6757593, -27.6927414, 27.7026138
34: -45.2508316, -13.7396774, -45.2207336, -13.7681198, -24.4148026, 24.3892212
35: -32.3239594, -2.8879237, -32.2924767, -2.9003372, -23.2695580, 23.2282333
36: -29.4710598, 2.3356924, -29.4142456, 2.3328547, -25.3449631, 25.2791748
37: -46.4435921, -5.4731498, -46.3789978, -5.4760394, -36.1531677, 36.0465012
38: -40.1170197, -2.7462893, -40.0529022, -2.7488847, -32.7423325, 32.6693649
39: -50.3528595, -7.9058075, -50.3057251, -7.9469137, -29.3671951, 29.3632965
40: -48.0190659, -17.6571712, -47.9897842, -17.6613140, -24.9808960, 24.9369202
41: -28.7951927, 0.9402766, -28.7549305, 0.9335394, -25.7377396, 25.6956177
42: -32.5019531, -9.4837532, -32.4853325, -9.4936075, -18.8468628, 18.8312454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=238, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1510

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1736

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5857867, upper bound: 19.6172589
time: 34.69 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6143907, upper bound: 19.6873254
time: 32.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 69.12 seconds
IS_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 69.12
Output dim: 26, lower bound: -19.5857867, upper bound: 19.6172589
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 69.12
Output dim: 26, lower bound: -19.6143907, upper bound: 19.6873254

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -33.9176102, -0.5721281, -33.9105339, -0.6101174, -24.7008514, 24.7277031
1: -13.1256657, 7.4556074, -13.1232901, 7.4325876, -16.0028076, 15.9980202
2: -12.1570501, 8.6559772, -12.1561489, 8.6223269, -15.4995270, 15.5041504
3: -26.8819695, -2.1730385, -26.8867531, -2.2239571, -20.0241699, 20.0424995
4: -16.8489704, 11.0629349, -16.8453941, 11.0238218, -20.9881973, 21.0518150
5: -21.6194286, 3.2957106, -21.6206055, 3.2584229, -19.3532791, 19.3522453
6: -34.7114983, -7.5547123, -34.6671066, -7.5559573, -22.5596085, 22.5277176
7: -20.9187927, 6.2748213, -20.9171829, 6.2381568, -21.4880676, 21.4687271
8: -31.0320148, 5.0197582, -31.0314407, 4.9837470, -26.4120026, 26.3846054
9: -18.9766064, 8.0436325, -18.9748363, 7.9792624, -23.2795868, 23.3478775
10: -16.7075405, 10.9891653, -16.6918068, 10.9147940, -25.3612137, 25.4220276
11: -5.9486456, 16.3725929, -5.9086189, 16.3708382, -17.6752968, 17.6194305
12: -22.6671314, 13.7048016, -22.6189404, 13.7060280, -29.2854004, 29.2636261
13: -33.4620209, 6.7661223, -33.4470634, 6.7315040, -30.2541809, 30.2412338
14: -37.0209618, 8.4331312, -36.9704208, 8.3957815, -43.4747772, 43.4382019
15: -17.1751328, 9.4590368, -17.1708355, 9.4065380, -24.0420685, 24.0940056
16: -19.6950378, 3.7900090, -19.6715584, 3.7505119, -20.9998322, 21.0070152
17: -26.4942932, 7.7561655, -26.4293251, 7.7575932, -34.2518845, 34.1854897
18: -7.6410713, 25.4296360, -7.5923176, 25.4278145, -31.1091614, 31.0759506
19: -1.0121088, 16.0883522, -0.9809856, 16.0862389, -15.5548897, 15.5327702
20: -7.0635900, 12.3395452, -7.0407391, 12.3313370, -18.4952393, 18.4989319
21: -5.5471010, 16.3083611, -5.5185423, 16.2948132, -20.9739075, 20.9742775
22: -2.6592288, 16.8847122, -2.6389647, 16.8758316, -16.6325760, 16.6441517
23: -4.0427074, 17.7237434, -4.0125122, 17.7214127, -18.8063431, 18.7515907
24: -2.8000278, 22.2497845, -2.7617426, 22.2480850, -21.6906891, 21.6542702
25: -5.3128648, 18.3320961, -5.2791409, 18.3251991, -21.0366821, 21.0025253
26: -7.8738585, 24.5517101, -7.8068504, 24.5397224, -30.4650116, 30.4098129
27: -5.9731970, 18.1599426, -5.9320059, 18.1604652, -20.6373291, 20.5952148
28: -2.9402342, 20.5416832, -2.9052539, 20.5389519, -21.6527405, 21.5922165
29: -2.4990463, 17.2151756, -2.4546728, 17.2063599, -15.6368713, 15.5961800
30: -9.8511105, 18.7338753, -9.8212700, 18.7305012, -26.2851410, 26.2621689
31: -5.5019169, 17.6785812, -5.4717722, 17.6708603, -21.6109924, 21.6116676
32: -28.7329369, -1.3305917, -28.7105007, -1.3557582, -20.9679108, 20.9812164
33: -50.8601608, -11.6288948, -50.8249588, -11.6761131, -27.6563263, 27.7006378
34: -45.2478790, -13.7405376, -45.2188530, -13.7686138, -24.4024277, 24.3859863
35: -32.3203735, -2.8882658, -32.2903519, -2.9005575, -23.2520180, 23.2230606
36: -29.4676189, 2.3335752, -29.4124470, 2.3317447, -25.3396149, 25.2815628
37: -46.4358444, -5.4734101, -46.3750381, -5.4762039, -36.1047211, 36.0423965
38: -40.1156082, -2.7485151, -40.0522041, -2.7500806, -32.7273712, 32.6948547
39: -50.3501740, -7.9062829, -50.3043518, -7.9471679, -29.3581696, 29.3726959
40: -48.0142441, -17.6575127, -47.9873428, -17.6614685, -24.9690323, 24.9341202
41: -28.7917557, 0.9394889, -28.7532063, 0.9331460, -25.7269516, 25.6931381
42: -32.5006371, -9.4850702, -32.4846458, -9.4943056, -18.8441849, 18.8290710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=238, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1013
type: A, layer: 1, pos: 1013
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1389
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1389
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1510

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1709

## Relational analysis of IS_A2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6302154, upper bound: 19.6728575
time: 40.74 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6302154, upper bound: 19.6860140
time: 33.27 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 76.12 seconds
IS_A2_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 76.12
Output dim: 26, lower bound: -19.6302154, upper bound: 19.6728575
IS_A2_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 76.12
Output dim: 26, lower bound: -19.6302154, upper bound: 19.6860140

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 47.98 + 753.38 = 801.36 seconds
