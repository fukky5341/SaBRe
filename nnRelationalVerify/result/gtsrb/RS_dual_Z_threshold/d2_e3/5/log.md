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
execution time: IAR + RelationalAnalysis = 2.45 + 45.02 = 47.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 26, lower bound: -19.7062175, upper bound: 19.7062176

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7036992, upper bound: 19.6420136
time: 38.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6420136, upper bound: 19.7036993
time: 44.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 83.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 83.83
Output dim: 26, lower bound: -19.7036992, upper bound: 19.6420136
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 83.83
Output dim: 26, lower bound: -19.6420136, upper bound: 19.7036993

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6886597, 24.6986084
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0163345, 16.0177307
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4863472, 15.4927521
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0981064, 20.0982666
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.9453354, 20.9605141
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3853149, 19.3859291
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6571350, 22.6547928
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5083313, 21.5101013
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4616013, 26.4673843
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3620834, 23.3664551
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4629288, 25.4572067
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5895309, 17.5725250
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3142853, 29.3080368
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3120728, 30.3194962
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5259399, 43.5252838
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1225433, 24.1288452
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0136108, 21.0137978
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1954498, 31.1938858
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5972195, 15.5968437
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5611839, 18.5560532
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0028801, 20.9949646
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.7160568, 16.7151794
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7943535, 18.7916641
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7365837, 21.7364731
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0592041, 21.0579262
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4876709, 30.4838257
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6912270, 20.6917648
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6736298, 21.6709671
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6376762, 15.6337166
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2539062, 26.2420654
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6652870, 21.6656418
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0608063, 21.0609818
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7463112, 27.7604752
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4708939, 24.4694405
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2797928, 23.2788239
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3914185, 25.3934021
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1749115, 36.1851196
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8328629, 32.8345566
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.3943481, 29.4170227
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0055313, 25.0133362
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7992249, 25.7999039
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8772354, 18.8792076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7027963, upper bound: 19.6070818
time: 30.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6688386, upper bound: 19.6411123
time: 40.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6986084, 24.6886597
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0177307, 16.0163345
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4927483, 15.4863472
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0982666, 20.0981102
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.9605179, 20.9453354
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3859253, 19.3853111
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6547928, 22.6571350
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5101013, 21.5083313
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4673843, 26.4616013
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3664551, 23.3620834
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4572067, 25.4629288
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5725250, 17.5895309
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3080444, 29.3142853
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3194962, 30.3120728
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5252838, 43.5259399
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1288452, 24.1225433
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0137939, 21.0136070
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1938934, 31.1954498
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5968418, 15.5972214
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5560570, 18.5611877
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9949608, 21.0028801
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.7151794, 16.7160568
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7916679, 18.7943497
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7364693, 21.7365875
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0579224, 21.0592003
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4838257, 30.4876785
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6917686, 20.6912231
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6709671, 21.6736298
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6337166, 15.6376762
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2420654, 26.2539139
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6656380, 21.6652832
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0609818, 21.0608063
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7604790, 27.7463150
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4694366, 24.4708939
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2788239, 23.2797890
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3934021, 25.3914185
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1851349, 36.1749039
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8345566, 32.8328705
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4170227, 29.3943481
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0133362, 25.0055313
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7999039, 25.7992249
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8792038, 18.8772354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6411122, upper bound: 19.6688386
time: 33.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6070817, upper bound: 19.7027964
time: 31.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 66.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 66.74
Output dim: 26, lower bound: -19.7027963, upper bound: 19.6070818
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 66.74
Output dim: 26, lower bound: -19.6688386, upper bound: 19.6411123
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 66.74
Output dim: 26, lower bound: -19.6411122, upper bound: 19.6688386
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 66.74
Output dim: 26, lower bound: -19.6070817, upper bound: 19.7027964

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6379471, 24.6549683
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9882812, 15.9941597
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4372482, 15.4513512
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0526733, 20.0602913
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8726349, 20.8992958
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3546295, 19.3601227
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6534882, 22.6513290
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4492035, 21.4607735
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3752060, 26.3950348
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3247681, 23.3348846
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4464188, 25.4428329
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5820541, 17.5641441
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2887192, 29.2761002
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2805786, 30.2913589
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5111389, 43.5126190
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1110458, 24.1184006
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9782715, 20.9854240
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1664886, 31.1591415
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5885639, 15.5865097
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5523872, 18.5455170
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9814301, 20.9693298
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6828804, 16.6752090
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7637634, 18.7554665
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7153091, 21.7108345
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0326462, 21.0228729
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4751358, 30.4698486
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6777611, 20.6755714
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6445770, 21.6364403
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6023712, 15.5903511
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2227859, 26.2052002
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6583710, 21.6557732
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0725174, 21.0739822
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7715263, 27.7874146
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4743881, 24.4729652
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2787247, 23.2777061
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3925018, 25.3944855
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1935959, 36.2051697
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8312531, 32.8328247
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4441452, 29.4779739
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0658188, 25.0840836
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8152237, 25.8177185
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8755569, 18.8780556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1576

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7018584, upper bound: 19.5880929
time: 34.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6838051, upper bound: 19.6061609
time: 35.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6549683, 24.6379471
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9941635, 15.9882812
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4513550, 15.4372482
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0602875, 20.0526810
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8992920, 20.8726349
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3601227, 19.3546333
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6513290, 22.6534843
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4607697, 21.4491997
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3950348, 26.3752060
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3348846, 23.3247681
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4428329, 25.4464188
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5641403, 17.5820503
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2761002, 29.2887192
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2913589, 30.2805710
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5126190, 43.5111389
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1184006, 24.1110458
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9854279, 20.9782677
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1591339, 31.1664886
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5865078, 15.5885658
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5455132, 18.5523834
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9693298, 20.9814262
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6752090, 16.6828823
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7554626, 18.7637672
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7108307, 21.7153091
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0228653, 21.0326538
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4698410, 30.4751434
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6755638, 20.6777611
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6364441, 21.6445808
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5903511, 15.6023712
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2052002, 26.2227859
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6557770, 21.6583710
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0739822, 21.0725174
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7874184, 27.7715302
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4729614, 24.4743881
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2777100, 23.2787247
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3944855, 25.3925018
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.2051773, 36.1935959
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8328400, 32.8312531
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4779739, 29.4441452
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0840836, 25.0658188
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8177185, 25.8152237
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8780594, 18.8755569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1576

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6061609, upper bound: 19.6838052
time: 39.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5880928, upper bound: 19.7018584
time: 40.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 81.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 81.87
Output dim: 26, lower bound: -19.7018584, upper bound: 19.5880929
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 81.87
Output dim: 26, lower bound: -19.6838051, upper bound: 19.6061609
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 81.87
Output dim: 26, lower bound: -19.6061609, upper bound: 19.6838052
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 81.87
Output dim: 26, lower bound: -19.5880928, upper bound: 19.7018584

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6361237, 24.6529465
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9873466, 15.9932594
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4360085, 15.4501648
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0452957, 20.0540848
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8649025, 20.8929138
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3494568, 19.3555946
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6532440, 22.6509628
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4465103, 21.4585228
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3747559, 26.3945541
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3220291, 23.3323822
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4470444, 25.4436646
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5808182, 17.5628204
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2877884, 29.2754364
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2645035, 30.2778931
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5147247, 43.5175323
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1107788, 24.1181030
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9782410, 20.9853935
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1626892, 31.1546021
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5889435, 15.5868912
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5523186, 18.5454521
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9792442, 20.9680710
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6844101, 16.6769524
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7576752, 18.7481880
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7116051, 21.7067642
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0327339, 21.0229645
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4732971, 30.4678040
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6701927, 20.6666908
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6398544, 21.6307793
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6001396, 15.5878067
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2199860, 26.2022552
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6597404, 21.6573143
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0725403, 21.0740089
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7710342, 27.7870026
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4739304, 24.4725037
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2685661, 23.2691498
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3867645, 25.3896408
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1859436, 36.1967621
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8243866, 32.8270798
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4318924, 29.4677200
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0538788, 25.0712738
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8148880, 25.8167725
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8700104, 18.8713913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7011865, upper bound: 19.5735307
time: 40.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6875396, upper bound: 19.5873340
time: 39.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6529465, 24.6361237
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9932594, 15.9873466
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4501686, 15.4360123
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0540848, 20.0452957
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8929176, 20.8649025
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3555908, 19.3494568
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6509628, 22.6532440
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4585190, 21.4465065
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3945541, 26.3747559
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3323822, 23.3220291
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4436646, 25.4470444
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5628204, 17.5808182
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2754288, 29.2877884
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2778931, 30.2644958
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5175323, 43.5147247
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1181030, 24.1107788
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9853897, 20.9782410
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1546021, 31.1626892
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5868912, 15.5889435
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5454445, 18.5523186
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9680672, 20.9792404
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6769524, 16.6844101
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7481918, 18.7576714
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7067604, 21.7116051
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0229683, 21.0327301
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4678040, 30.4732971
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6666908, 20.6701889
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6307831, 21.6398544
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5878067, 15.6001396
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2022476, 26.2199860
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6573143, 21.6597404
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0740128, 21.0725403
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7870026, 27.7710342
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4725037, 24.4739304
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2691460, 23.2685699
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3896408, 25.3867645
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1967621, 36.1859436
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8270721, 32.8243866
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4677200, 29.4318924
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0712738, 25.0538788
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8167725, 25.8148880
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8713913, 18.8700066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5873340, upper bound: 19.6875396
time: 37.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5735306, upper bound: 19.7011866
time: 36.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 76.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 76.10
Output dim: 26, lower bound: -19.7011865, upper bound: 19.5735307
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 76.10
Output dim: 26, lower bound: -19.6875396, upper bound: 19.5873340
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 76.10
Output dim: 26, lower bound: -19.5873340, upper bound: 19.6875396
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 76.10
Output dim: 26, lower bound: -19.5735306, upper bound: 19.7011866

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6276703, 24.6407547
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9877434, 15.9928741
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4353828, 15.4487839
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0409927, 20.0512047
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8634071, 20.8917656
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3463287, 19.3530350
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6505356, 22.6485062
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4463501, 21.4583397
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3728180, 26.3916473
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3213348, 23.3312454
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4463348, 25.4437714
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5807724, 17.5627747
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2833328, 29.2725983
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2587738, 30.2739334
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5141907, 43.5183258
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1085968, 24.1152649
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9769287, 20.9833641
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1604996, 31.1517715
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5889111, 15.5868607
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5519905, 18.5454788
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9773827, 20.9678192
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6836815, 16.6774273
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7531319, 18.7415047
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7096710, 21.7044067
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0323219, 21.0228348
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4719238, 30.4661102
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6665497, 20.6613083
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6381073, 21.6282310
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6000328, 15.5878754
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2190475, 26.2023697
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6592216, 21.6572647
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0704117, 21.0721054
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7705574, 27.7865753
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4728622, 24.4721947
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2631111, 23.2659416
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3844299, 25.3880539
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1786804, 36.1866455
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8228455, 32.8258820
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4307251, 29.4667511
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0477829, 25.0632553
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8132172, 25.8137207
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8677177, 18.8668213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 888

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6386134, upper bound: 19.5732476
time: 30.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7007425, upper bound: 19.5156110
time: 38.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6239319, 24.6444588
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9869576, 15.9936371
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4346275, 15.4495430
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0424271, 20.0497856
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8637581, 20.8914185
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3469009, 19.3524628
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6507874, 22.6482582
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4463196, 21.4583473
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3718491, 26.3926086
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3208923, 23.3317184
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4471512, 25.4429512
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5807724, 17.5627747
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2849045, 29.2709808
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2605362, 30.2721634
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5155334, 43.5169983
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1079407, 24.1159172
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9762115, 20.9840813
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1598587, 31.1524200
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5889149, 15.5868568
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5523415, 18.5451241
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9789925, 20.9662132
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6848869, 16.6762199
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7509880, 18.7436485
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7092438, 21.7047386
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0325966, 21.0225525
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4716034, 30.4664383
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6648026, 20.6630478
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6373062, 21.6290512
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6002083, 15.5876999
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2200699, 26.2013092
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6596870, 21.6567955
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0706406, 21.0718842
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7706032, 27.7865295
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4736252, 24.4714355
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2653923, 23.2636871
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3851776, 25.3873062
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1758423, 36.1894531
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8232269, 32.8255310
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4309311, 29.4665451
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0458603, 25.0651550
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8118439, 25.8150024
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8654366, 18.8689957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 888

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6249710, upper bound: 19.5870550
time: 36.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6870990, upper bound: 19.5294009
time: 53.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6444626, 24.6239319
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9936333, 15.9869614
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4495430, 15.4346275
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0497818, 20.0424309
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8914223, 20.8637543
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3524628, 19.3469009
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6482544, 22.6507835
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4583435, 21.4463234
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3926086, 26.3718491
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3317184, 23.3208923
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4429474, 25.4471512
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5627747, 17.5807724
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2709732, 29.2849121
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2721634, 30.2605362
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5169983, 43.5155334
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1159134, 24.1079407
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9840851, 20.9762154
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1524124, 31.1598587
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5868549, 15.5889149
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5451241, 18.5523453
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9662132, 20.9789925
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6762199, 16.6848869
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7436485, 18.7509880
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7047424, 21.7092476
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0225563, 21.0326004
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4664307, 30.4716034
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6630478, 20.6648064
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6290512, 21.6373062
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5876999, 15.6002083
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2013092, 26.2200699
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6567955, 21.6596909
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0718842, 21.0706406
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7865257, 27.7706070
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4714355, 24.4736214
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2636909, 23.2653961
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3873062, 25.3851776
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1894531, 36.1758347
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8255310, 32.8232269
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4665527, 29.4309311
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0651550, 25.0458603
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8150024, 25.8118439
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8689995, 18.8654404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 888

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5294008, upper bound: 19.6870990
time: 36.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5870550, upper bound: 19.6249710
time: 36.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6407547, 24.6276703
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9928780, 15.9877434
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4487801, 15.4353867
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0512009, 20.0409966
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8917656, 20.8634109
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3530350, 19.3463249
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6485062, 22.6505356
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4583435, 21.4463539
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3916473, 26.3728180
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3312454, 23.3213348
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4437714, 25.4463310
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5627747, 17.5807724
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2726059, 29.2833328
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2739334, 30.2587738
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5183258, 43.5141907
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1152649, 24.1085968
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9833679, 20.9769325
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1517715, 31.1605072
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5868626, 15.5889130
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5454750, 18.5519905
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9678154, 20.9773865
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6774292, 16.6836796
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7415047, 18.7531281
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7044067, 21.7096748
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0228310, 21.0323181
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4661102, 30.4719315
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6613083, 20.6665497
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6282349, 21.6381073
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5878754, 15.6000328
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2023697, 26.2190475
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6572609, 21.6592216
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0721054, 21.0704155
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7865715, 27.7705612
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4721985, 24.4728622
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2659416, 23.2631111
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3880539, 25.3844299
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1866455, 36.1786880
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8258820, 32.8228455
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4667511, 29.4307175
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0632553, 25.0477829
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8137207, 25.8132172
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8668251, 18.8677216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 888

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5156109, upper bound: 19.7007425
time: 35.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5732476, upper bound: 19.6386134
time: 46.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 84.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.6386134, upper bound: 19.5732476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.7007425, upper bound: 19.5156110
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.6249710, upper bound: 19.5870550
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.6870990, upper bound: 19.5294009
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.5294008, upper bound: 19.6870990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.5870550, upper bound: 19.6249710
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.5156109, upper bound: 19.7007425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 84.60
Output dim: 26, lower bound: -19.5732476, upper bound: 19.6386134

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6052475, 24.6231537
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9815178, 15.9894867
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4123383, 15.4288025
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0290375, 20.0408440
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8429718, 20.8743210
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3280411, 19.3367577
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6483917, 22.6457596
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4161987, 21.4326859
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3697357, 26.3928986
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3135529, 23.3248901
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4425430, 25.4388618
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5824547, 17.5639725
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2585983, 29.2443924
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2582474, 30.2734451
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5160217, 43.5200958
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1085358, 24.1154480
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9727020, 20.9800758
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1597824, 31.1509018
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5899811, 15.5874844
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5478668, 18.5405579
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9784012, 20.9674644
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6640625, 16.6573772
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7516251, 18.7398949
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7076645, 21.7022781
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0216560, 21.0102501
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4731369, 30.4670715
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6669540, 20.6618881
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6355515, 21.6239166
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5857925, 15.5708160
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2098389, 26.1905136
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6598701, 21.6576004
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0703125, 21.0720024
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7742386, 27.7920532
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4726028, 24.4702034
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2604904, 23.2625465
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3839951, 25.3876343
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1783295, 36.1863632
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8229141, 32.8262558
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4221191, 29.4597397
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0415039, 25.0595703
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8079376, 25.8091049
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8694916, 18.8698616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6958180, upper bound: 19.4996002
time: 31.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6748227, upper bound: 19.5090858
time: 37.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6015091, 24.6268578
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9807396, 15.9902496
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4115829, 15.4295654
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0304718, 20.0394287
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8433151, 20.8739777
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3286209, 19.3361855
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6486359, 22.6455116
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4161682, 21.4326935
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3687592, 26.3938599
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3131104, 23.3253555
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4433670, 25.4380417
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5824547, 17.5639725
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2601700, 29.2427750
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2600098, 30.2716827
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5173645, 43.5187683
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1078796, 24.1160965
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9719772, 20.9807930
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1591263, 31.1515503
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5899811, 15.5874805
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5482254, 18.5402031
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9800110, 20.9658585
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6652679, 16.6561699
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7494888, 18.7420387
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7072372, 21.7026062
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0219460, 21.0099678
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4728165, 30.4674072
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6652069, 20.6636314
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6347504, 21.6247368
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5859680, 15.5706444
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2108612, 26.1894608
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6603355, 21.6571312
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0705414, 21.0717773
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7742844, 27.7920074
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4733658, 24.4694443
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2627716, 23.2602921
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3847351, 25.3868866
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1754761, 36.1891708
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8232803, 32.8259048
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4223251, 29.4595337
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0395813, 25.0614624
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8065567, 25.8103790
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8672104, 18.8720322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6821155, upper bound: 19.5133911
time: 39.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6611353, upper bound: 19.5228942
time: 32.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6268539, 24.6015129
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9902458, 15.9807358
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4295654, 15.4115791
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0394287, 20.0304680
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8739777, 20.8433151
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3361816, 19.3286171
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6455154, 22.6486359
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4326935, 21.4161682
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3938599, 26.3687592
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3253555, 23.3131104
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4380417, 25.4433670
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5639763, 17.5824585
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2427750, 29.2601700
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2716827, 30.2600098
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5187683, 43.5173492
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1160965, 24.1078796
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9807968, 20.9719734
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1515427, 31.1591263
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5874786, 15.5899811
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5401993, 18.5482254
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9658585, 20.9800072
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6561699, 16.6652699
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7420425, 18.7494850
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7026062, 21.7072372
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0099678, 21.0219421
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4673996, 30.4728165
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6636353, 20.6652069
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6247330, 21.6347542
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5706406, 15.5859642
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.1894531, 26.2108612
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6571312, 21.6603394
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0717773, 21.0705376
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7920074, 27.7742844
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4694443, 24.4733620
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2602997, 23.2627716
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3868866, 25.3847351
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1891632, 36.1754684
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8259048, 32.8232880
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4595337, 29.4223251
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0614624, 25.0395813
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8103790, 25.8065567
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8720322, 18.8672066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5228942, upper bound: 19.6611354
time: 31.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5133910, upper bound: 19.6821155
time: 35.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6231537, 24.6052475
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9894829, 15.9815178
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4288025, 15.4123383
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0408478, 20.0290337
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8743210, 20.8429718
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3367615, 19.3280449
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6457596, 22.6483917
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4326859, 21.4161987
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3928986, 26.3697357
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3248901, 23.3135529
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4388657, 25.4425468
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5639763, 17.5824585
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2443924, 29.2585983
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2734451, 30.2582474
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5200958, 43.5160217
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1154480, 24.1085358
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9800720, 20.9726944
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1509018, 31.1597748
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5874863, 15.5899792
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5405579, 18.5478706
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9674606, 20.9784050
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6573792, 16.6640625
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7398987, 18.7516289
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7022781, 21.7076645
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0102577, 21.0216599
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4670792, 30.4731445
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6618881, 20.6669502
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6239166, 21.6355515
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5708160, 15.5857925
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.1905136, 26.2098389
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6575966, 21.6598701
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0720062, 21.0703125
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7920532, 27.7742386
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4701996, 24.4726028
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2625504, 23.2604866
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3876343, 25.3839951
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1863556, 36.1783218
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8262711, 32.8228989
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4597397, 29.4221191
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0595703, 25.0414963
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8091049, 25.8079376
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8698578, 18.8694916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5090858, upper bound: 19.6748228
time: 35.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.4996001, upper bound: 19.6958180
time: 34.06 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 71.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.6958180, upper bound: 19.4996002
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.6748227, upper bound: 19.5090858
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.6821155, upper bound: 19.5133911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.6611353, upper bound: 19.5228942
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.5228942, upper bound: 19.6611354
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.5133910, upper bound: 19.6821155
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.5090858, upper bound: 19.6748228
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 71.92
Output dim: 26, lower bound: -19.4996001, upper bound: 19.6958180

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6024017, 24.6222878
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9783058, 15.9885635
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4095383, 15.4282074
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0248260, 20.0396423
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8372650, 20.8731689
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3264694, 19.3363190
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6482010, 22.6467247
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4113541, 21.4316292
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3614120, 26.3905258
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3088379, 23.3237305
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4388657, 25.4378204
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5821037, 17.5632591
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2588501, 29.2438507
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2574387, 30.2725983
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5142517, 43.5196686
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1081314, 24.1152687
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9691467, 20.9809685
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1583862, 31.1458359
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5898819, 15.5860367
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5477295, 18.5404053
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9785767, 20.9658279
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6646652, 16.6550884
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7516785, 18.7358818
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7064972, 21.6982269
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0246429, 21.0091171
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4720840, 30.4655075
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6663742, 20.6585464
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6350632, 21.6219521
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5876427, 15.5698509
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2097168, 26.1901855
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6610756, 21.6560707
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0700531, 21.0717278
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7739334, 27.7914352
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4721069, 24.4695053
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2603836, 23.2621078
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3832626, 25.3858795
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1782684, 36.1863022
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8214874, 32.8227768
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4216690, 29.4600067
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0408401, 25.0643845
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8077316, 25.8092575
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8692703, 18.8698959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6952763, upper bound: 19.4956500
time: 35.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6901787, upper bound: 19.4985246
time: 37.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6222916, 24.6024055
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9885674, 15.9783058
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4282074, 15.4095383
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0396423, 20.0248222
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8731689, 20.8372650
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3363190, 19.3264656
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6467285, 22.6482010
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4316177, 21.4113579
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3905258, 26.3614120
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3237228, 23.3088379
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4378204, 25.4388657
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5632591, 17.5820999
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2438507, 29.2588425
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2725983, 30.2574387
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5196686, 43.5142517
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1152725, 24.1081352
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9809647, 20.9691429
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1458435, 31.1583939
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5860367, 15.5898819
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5404053, 18.5477295
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9658279, 20.9785767
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6550903, 16.6646652
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7358780, 18.7516747
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.6982269, 21.7064972
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0091095, 21.0246429
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4655075, 30.4720917
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6585464, 20.6663704
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6219559, 21.6350670
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5698509, 15.5876427
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.1901779, 26.2097168
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6560707, 21.6610718
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0717316, 21.0700493
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7914352, 27.7739334
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4695129, 24.4721069
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2621078, 23.2603836
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3858795, 25.3832626
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1862946, 36.1782608
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8227692, 32.8214874
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4600067, 29.4216690
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0643845, 25.0408401
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.8092575, 25.8077316
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8698959, 18.8692665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1560

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.4985246, upper bound: 19.6901787
time: 36.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.4956500, upper bound: 19.6952764
time: 32.94 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 71.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 71.52
Output dim: 26, lower bound: -19.6952763, upper bound: 19.4956500
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 71.52
Output dim: 26, lower bound: -19.6901787, upper bound: 19.4985246
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 71.52
Output dim: 26, lower bound: -19.4985246, upper bound: 19.6901787
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 71.52
Output dim: 26, lower bound: -19.4956500, upper bound: 19.6952764

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.5961761, 24.6150360
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9757462, 15.9862518
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4070778, 15.4260902
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0185699, 20.0341911
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8208618, 20.8594246
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3211212, 19.3317299
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6341324, 22.6300468
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4066238, 21.4277267
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3607101, 26.3898010
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3064194, 23.3212585
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4409943, 25.4400406
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5815964, 17.5627670
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2585831, 29.2435608
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2385483, 30.2566681
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5138245, 43.5216370
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1074448, 24.1145401
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9684525, 20.9800224
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1567917, 31.1441422
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5866947, 15.5838089
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5452881, 18.5384674
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9651108, 20.9545326
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6637840, 16.6554165
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7437401, 18.7262421
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7059937, 21.6977615
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0236969, 21.0083809
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4730759, 30.4665451
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6558838, 20.6465797
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6328278, 21.6191444
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5876694, 15.5698814
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2095795, 26.1900864
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6602402, 21.6564598
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0658340, 21.0661850
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7649231, 27.7811661
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4646301, 24.4605865
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2547379, 23.2579651
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3812943, 25.3843079
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1584320, 36.1626434
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8208694, 32.8226776
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4163895, 29.4554901
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0144119, 25.0328522
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7901230, 25.7882462
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8428383, 18.8383636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6895282, upper bound: 19.4874208
time: 33.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6828947, upper bound: 19.4913417
time: 41.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.5951462, 24.6160126
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9759521, 15.9860001
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4074059, 15.4257469
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0193024, 20.0333900
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8235168, 20.8567696
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3218231, 19.3309746
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6315231, 22.6326370
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4074249, 21.4269028
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3606720, 26.3898239
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3063736, 23.3213043
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4410858, 25.4399490
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5816116, 17.5627556
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2585526, 29.2435684
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2415009, 30.2537079
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5162201, 43.5192566
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1074066, 24.1145782
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9681931, 20.9802742
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1567001, 31.1442337
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5876331, 15.5828495
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5457916, 18.5379601
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9672775, 20.9523697
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6649933, 16.6542072
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7420464, 18.7279434
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7060318, 21.6977005
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0239105, 21.0081635
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4731369, 30.4664841
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6544037, 20.6480598
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6322632, 21.6196404
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5876732, 15.5698738
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2096252, 26.1900406
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6614609, 21.6552391
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0645065, 21.0675087
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7636642, 27.7823944
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4631882, 24.4620247
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2562408, 23.2564621
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3816605, 25.3839035
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1546021, 36.1664734
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8213577, 32.8221741
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4170990, 29.4547195
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0093155, 25.0379562
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7867203, 25.7916489
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8377342, 18.8434677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6844382, upper bound: 19.4902862
time: 37.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6777981, upper bound: 19.4942040
time: 31.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6160049, 24.5951500
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9860001, 15.9759521
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4257469, 15.4074059
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0333862, 20.0193062
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8567657, 20.8235207
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3309784, 19.3218193
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6326370, 22.6315193
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4268951, 21.4074249
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3898239, 26.3606720
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3213043, 23.3063736
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4399490, 25.4410858
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5627518, 17.5816078
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2435684, 29.2585526
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2537079, 30.2415009
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5192566, 43.5162201
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1145782, 24.1074066
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9802780, 20.9681969
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1442337, 31.1567001
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5828495, 15.5876331
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5379562, 18.5457916
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9523697, 20.9672813
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6542053, 16.6649933
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7279472, 18.7420387
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.6977005, 21.7060318
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0081635, 21.0239067
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4664841, 30.4731293
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6480560, 20.6543999
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6196442, 21.6322594
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5698738, 15.5876732
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.1900406, 26.2096252
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6552353, 21.6614647
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0675125, 21.0645027
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7823944, 27.7636642
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4620209, 24.4631882
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2564621, 23.2562370
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3839035, 25.3816605
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1664734, 36.1546021
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8221817, 32.8213501
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4547195, 29.4170990
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0379562, 25.0093155
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7916489, 25.7867203
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8434639, 18.8377380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4942040, upper bound: 19.6777981
time: 41.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4902862, upper bound: 19.6844382
time: 33.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6150284, 24.5961800
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9862518, 15.9757462
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4260902, 15.4070778
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0341949, 20.0185661
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8594208, 20.8208656
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3317261, 19.3211212
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6300430, 22.6341324
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4277267, 21.4066238
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3898010, 26.3607101
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3212585, 23.3064194
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4400406, 25.4409943
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5627670, 17.5815964
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2435532, 29.2585754
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2566681, 30.2385483
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5216370, 43.5138245
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1145401, 24.1074448
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9800186, 20.9684486
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1441422, 31.1567993
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5838070, 15.5866966
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5384674, 18.5452843
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9545364, 20.9651146
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6554184, 16.6637840
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7262383, 18.7437401
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.6977615, 21.7059937
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0083771, 21.0236931
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4665451, 30.4730682
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6465759, 20.6558800
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6191406, 21.6328239
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5698814, 15.5876694
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.1900864, 26.2095795
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6564560, 21.6602364
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0661850, 21.0658302
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7811661, 27.7649307
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4605865, 24.4646301
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2579651, 23.2547379
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3843079, 25.3812943
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1626434, 36.1584320
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8226700, 32.8208771
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4554977, 29.4163895
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0328522, 25.0144119
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7882462, 25.7901230
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8383675, 18.8428383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4913416, upper bound: 19.6828948
time: 39.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.4874207, upper bound: 19.6895283
time: 37.26 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 78.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.6895282, upper bound: 19.4874208
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.6828947, upper bound: 19.4913417
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.6844382, upper bound: 19.4902862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.6777981, upper bound: 19.4942040
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.4942040, upper bound: 19.6777981
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.4902862, upper bound: 19.6844382
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.4913416, upper bound: 19.6828948
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 78.38
Output dim: 26, lower bound: -19.4874207, upper bound: 19.6895283

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.5931168, 24.6172409
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9735031, 15.9878654
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4052238, 15.4281769
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0169373, 20.0355911
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8184128, 20.8616638
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3188782, 19.3334846
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6339111, 22.6299324
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4041290, 21.4295807
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3552399, 26.3937607
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3043518, 23.3227463
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4387360, 25.4416580
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5818367, 17.5624962
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2625504, 29.2429657
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2434082, 30.2564240
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5122223, 43.5227966
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1064148, 24.1152039
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9667053, 20.9839973
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1568146, 31.1440582
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5869579, 15.5837479
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5452194, 18.5393524
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9652100, 20.9545097
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6667976, 16.6551418
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7445831, 18.7256203
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7069016, 21.6967583
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0265808, 21.0082512
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4729309, 30.4679565
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6561356, 20.6464043
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6329651, 21.6190147
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5891838, 15.5698166
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2101898, 26.1894608
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6625252, 21.6562881
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0672913, 21.0640411
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7681122, 27.7772064
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4663239, 24.4581985
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2568359, 23.2550163
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3837967, 25.3808441
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1598434, 36.1606979
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8231735, 32.8197632
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4191666, 29.4510345
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0143738, 25.0332184
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7901993, 25.7876663
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8427963, 18.8384476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1591

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6722677, upper bound: 19.4530231
time: 43.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6518008, upper bound: 19.4711111
time: 30.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.6150284, 24.5931168
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -15.9862518, 15.9735031
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.4260902, 15.4052200
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0341949, 20.0169373
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -20.8594208, 20.8184090
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3317261, 19.3188858
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6299286, 22.6341324
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4277267, 21.4041290
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.3898010, 26.3552399
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3212585, 23.3043518
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4400406, 25.4387321
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.5624962, 17.5815964
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.2429581, 29.2585754
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.2564240, 30.2385483
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5216370, 43.5122375
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1145401, 24.1064186
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9800186, 20.9667091
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1440582, 31.1567993
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5837460, 15.5866966
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5384674, 18.5452194
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -20.9545135, 20.9651146
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6551399, 16.6637840
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.7256165, 18.7437401
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.6967545, 21.7059937
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0082550, 21.0236931
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.4665451, 30.4729385
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6464005, 20.6558800
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6190186, 21.6328239
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.5698166, 15.5876694
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.1894608, 26.2095795
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6562843, 21.6602364
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0640411, 21.0658302
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7772064, 27.7649307
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4581985, 24.4646301
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2550125, 23.2547379
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3808365, 25.3812943
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1606979, 36.1584320
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8197556, 32.8208771
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4510345, 29.4163895
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0328522, 25.0143738
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7876663, 25.7901230
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8383675, 18.8427963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1591

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4711111, upper bound: 19.6518008
time: 26.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4530231, upper bound: 19.6722678
time: 35.12 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 63.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 63.42
Output dim: 26, lower bound: -19.6722677, upper bound: 19.4530231
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 63.42
Output dim: 26, lower bound: -19.6518008, upper bound: 19.4711111
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 63.42
Output dim: 26, lower bound: -19.4711111, upper bound: 19.6518008
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 63.42
Output dim: 26, lower bound: -19.4530231, upper bound: 19.6722678

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 47.48 + 1735.15 = 1782.62 seconds
