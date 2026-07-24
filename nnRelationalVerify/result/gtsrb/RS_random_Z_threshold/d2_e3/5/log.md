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
execution time: IAR + RelationalAnalysis = 2.44 + 45.35 = 47.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 26, lower bound: -19.7062175, upper bound: 19.7062176

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1583

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1474

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7053697, upper bound: 19.7010215
time: 36.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7010214, upper bound: 19.7053697
time: 41.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 78.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 78.07
Output dim: 26, lower bound: -19.7053697, upper bound: 19.7010215
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 78.07
Output dim: 26, lower bound: -19.7010214, upper bound: 19.7053697

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7232971, 24.7243614
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0162354, 16.0167007
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5230560, 15.5237656
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0926437, 20.0931244
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0305252, 21.0315704
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3876343, 19.3880920
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6165237, 22.6159439
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5186462, 21.5190582
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4446487, 26.4458389
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3785400, 23.3791656
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4869156, 25.4874191
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6672974, 17.6668358
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3655243, 29.3646774
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3650818, 30.3650360
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5321503, 43.5325165
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1069717, 24.1076317
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0140991, 21.0145416
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1990967, 31.1989975
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5995064, 15.5995064
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5373077, 18.5370674
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0122185, 21.0119476
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6932449, 16.6931973
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8158684, 18.8157539
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7383881, 21.7382393
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0644760, 21.0640945
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5108109, 30.5108490
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6914520, 20.6911545
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6700974, 21.6698303
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6591873, 15.6588745
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2910156, 26.2903595
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6657562, 21.6655121
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0538025, 21.0534744
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8263550, 27.8262329
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4646378, 24.4645386
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2889977, 23.2889061
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4070816, 25.4066162
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1814499, 36.1815033
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8241425, 32.8237228
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5160751, 29.5160751
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0442352, 25.0442200
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7938843, 25.7937775
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9028053, 18.9025421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1315

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1390

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6988818, upper bound: 19.6947562
time: 36.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6991100, upper bound: 19.6945281
time: 36.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7243652, 24.7232971
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0167084, 16.0162354
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5237656, 15.5230560
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0931244, 20.0926437
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0315704, 21.0305252
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3880920, 19.3876343
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6159439, 22.6165276
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5190582, 21.5186462
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4458389, 26.4446487
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3791656, 23.3785400
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4874191, 25.4869156
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6668320, 17.6672974
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3646698, 29.3655243
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3650360, 30.3650818
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5325165, 43.5321655
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1076279, 24.1069717
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0145416, 21.0141029
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1990051, 31.1991043
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5995064, 15.5995064
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5370712, 18.5373077
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0119438, 21.0122147
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6931992, 16.6932430
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8157539, 18.8158684
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7382355, 21.7383919
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0640945, 21.0644760
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5108414, 30.5108185
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6911545, 20.6914482
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6698303, 21.6700974
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6588745, 15.6591873
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2903595, 26.2910233
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6655121, 21.6657486
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0534744, 21.0538025
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8262329, 27.8263550
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4645386, 24.4646378
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2889061, 23.2889938
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4066162, 25.4070892
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1814957, 36.1814499
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8237305, 32.8241425
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5160751, 29.5160751
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0442200, 25.0442352
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7937775, 25.7938843
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9025383, 18.9028053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1558

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6960225, upper bound: 19.6945998
time: 36.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6902493, upper bound: 19.7003751
time: 26.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 65.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 65.10
Output dim: 26, lower bound: -19.6988818, upper bound: 19.6947562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 65.10
Output dim: 26, lower bound: -19.6991100, upper bound: 19.6945281
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 65.10
Output dim: 26, lower bound: -19.6960225, upper bound: 19.6945998
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 65.10
Output dim: 26, lower bound: -19.6902493, upper bound: 19.7003751

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7235489, 24.7240334
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0164642, 16.0165367
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5232010, 15.5236130
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0928650, 20.0928726
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0306854, 21.0313644
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3879623, 19.3878670
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6164474, 22.6157913
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5188217, 21.5188217
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4449158, 26.4454956
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3786926, 23.3789444
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4869308, 25.4874191
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6671677, 17.6669922
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3654327, 29.3646622
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3652725, 30.3647766
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5323639, 43.5324249
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1070023, 24.1075897
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0142822, 21.0143929
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1988983, 31.1991501
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5993805, 15.5996056
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5372849, 18.5370750
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0121117, 21.0119324
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6930809, 16.6933289
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8157234, 18.8158684
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7382050, 21.7383995
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0643387, 21.0641937
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5106964, 30.5109406
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6913681, 20.6912155
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6698914, 21.6699829
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6589699, 15.6590271
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2907562, 26.2905121
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6656342, 21.6656227
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0537720, 21.0533409
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8264427, 27.8260651
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4646378, 24.4645462
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2889824, 23.2888947
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4071121, 25.4065094
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1815033, 36.1814423
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8243332, 32.8234711
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5162735, 29.5157013
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0443497, 25.0440216
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7939301, 25.7937012
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9029579, 18.9023132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6905358, upper bound: 19.6893263
time: 31.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6934525, upper bound: 19.6864094
time: 36.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7229691, 24.7243614
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0160675, 16.0167007
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5229034, 15.5237656
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0923920, 20.0931244
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0303116, 21.0315704
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3874130, 19.3880920
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6163788, 22.6159439
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5184097, 21.5190582
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4443130, 26.4458389
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3783188, 23.3791656
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4869156, 25.4874191
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6672974, 17.6667061
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3655243, 29.3645859
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3648224, 30.3650360
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5320740, 43.5325165
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1069260, 24.1076317
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0139542, 21.0145416
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1990967, 31.1987991
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5995064, 15.5993805
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5373077, 18.5370522
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0122185, 21.0118484
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6932449, 16.6930351
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8158684, 18.8156166
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7383881, 21.7380562
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0644760, 21.0639572
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5108109, 30.5107346
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6914520, 20.6910744
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6700974, 21.6696243
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6591873, 15.6586533
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2910156, 26.2900925
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6657562, 21.6653938
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0536652, 21.0534744
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8261833, 27.8262329
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4646378, 24.4645386
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2889977, 23.2888947
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4069748, 25.4066162
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1813965, 36.1815033
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8238907, 32.8237228
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5157089, 29.5160751
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0440445, 25.0442200
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7938080, 25.7937775
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9025764, 18.9025421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1334

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1491

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6973089, upper bound: 19.6749533
time: 37.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6800355, upper bound: 19.6927267
time: 39.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7213058, 24.7255096
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0144653, 16.0178452
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5218811, 15.5251160
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0914917, 20.0940437
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0291214, 21.0327644
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3858566, 19.3893890
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6157303, 22.6164207
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5165558, 21.5204926
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4403610, 26.4486008
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3771057, 23.3800278
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4851608, 25.4885368
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6670227, 17.6670113
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3686600, 29.3649368
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3699112, 30.3648453
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5309143, 43.5332642
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1066055, 24.1076126
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0127487, 21.0180206
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1990280, 31.1990356
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5997734, 15.5994492
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5369949, 18.5381927
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0120316, 21.0121994
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6961098, 16.6928692
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8165894, 18.8152275
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7391434, 21.7373924
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0669861, 21.0643387
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5107117, 30.5122147
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6913338, 20.6912727
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6699677, 21.6699677
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6603851, 15.6591187
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2909546, 26.2904129
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6676979, 21.6654739
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0549316, 21.0516510
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8294067, 27.8223953
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4662476, 24.4622650
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2910042, 23.2860451
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4091263, 25.4036179
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1828918, 36.1794815
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8260040, 32.8212051
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5188446, 29.5116119
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0441895, 25.0445633
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7938538, 25.7933044
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9025002, 18.9028854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1297

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6936534, upper bound: 19.6938840
time: 35.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6953041, upper bound: 19.6922319
time: 40.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7243652, 24.7202415
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0167084, 16.0139923
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5237656, 15.5211716
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0931244, 20.0910110
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0315704, 21.0280685
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3880920, 19.3853989
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6158371, 22.6165276
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5190582, 21.5161362
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4458389, 26.4391785
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3791656, 23.3764801
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4874191, 25.4846573
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6665497, 17.6672974
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3640823, 29.3655243
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3647995, 30.3650818
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5325165, 43.5305481
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1076279, 24.1059494
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0145416, 21.0123062
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1989365, 31.1991043
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5994492, 15.5995064
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5370712, 18.5372391
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0119324, 21.0122147
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6928215, 16.6932430
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8151169, 18.8158684
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7372360, 21.7383919
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0639648, 21.0644760
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5108414, 30.5106735
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6909828, 20.6914482
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6697006, 21.6700974
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6588058, 15.6591873
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2897491, 26.2910233
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6652412, 21.6657486
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0513229, 21.0538025
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8222656, 27.8263550
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4621658, 24.4646378
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2859612, 23.2889938
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4031525, 25.4070892
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1795349, 36.1814499
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8207855, 32.8241425
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5116119, 29.5160751
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0442200, 25.0442047
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7931976, 25.7938843
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9025383, 18.9027596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1348

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1397

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6901985, upper bound: 19.7003751
time: 33.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6902493, upper bound: 19.7003244
time: 38.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 74.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6905358, upper bound: 19.6893263
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6934525, upper bound: 19.6864094
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6973089, upper bound: 19.6749533
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6800355, upper bound: 19.6927267
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6936534, upper bound: 19.6938840
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6953041, upper bound: 19.6922319
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6901985, upper bound: 19.7003751
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 74.14
Output dim: 26, lower bound: -19.6902493, upper bound: 19.7003244

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7257690, 24.7216835
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0176430, 16.0153694
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5239868, 15.5227242
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0939178, 20.0926094
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0328178, 21.0310135
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3880997, 19.3878479
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6157227, 22.6170654
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5206146, 21.5183334
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4465179, 26.4453506
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3817978, 23.3781052
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4863586, 25.4878769
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6668930, 17.6671066
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3653107, 29.3650818
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3682709, 30.3615036
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5337830, 43.5320740
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1070557, 24.1075897
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0152206, 21.0135384
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1980286, 31.2000427
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5993118, 15.5997829
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5369644, 18.5372314
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0120506, 21.0119019
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6921959, 16.6940269
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8156090, 18.8164368
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7379227, 21.7397766
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0637589, 21.0650101
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5102463, 30.5112152
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6905136, 20.6917992
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6698227, 21.6717606
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6574440, 15.6602325
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2890472, 26.2923431
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6651611, 21.6659660
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0540161, 21.0530472
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8272552, 27.8246155
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4645462, 24.4660339
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2894020, 23.2880173
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4083557, 25.4048080
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1823730, 36.1801453
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8262329, 32.8209381
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5195618, 29.5119553
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0443420, 25.0438614
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7939072, 25.7935562
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9028969, 18.9021454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1302

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1334

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6904109, upper bound: 19.6890287
time: 43.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6902443, upper bound: 19.6891955
time: 37.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7211990, 24.7240334
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0152931, 16.0165367
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5223083, 15.5236130
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0926056, 20.0928726
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0303307, 21.0313644
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3879395, 19.3878670
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6164474, 22.6150665
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5183411, 21.5188217
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4447632, 26.4454956
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3778534, 23.3789444
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4869308, 25.4868469
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6671677, 17.6667175
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3654327, 29.3645325
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3619995, 30.3647766
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5320129, 43.5324249
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1070023, 24.1075897
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0134277, 21.0143929
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1988983, 31.1982880
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5993805, 15.5995369
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5372849, 18.5367508
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0120888, 21.0119324
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6930809, 16.6924477
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8157234, 18.8157501
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7382050, 21.7381172
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0643387, 21.0636177
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5106964, 30.5104828
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6913681, 20.6903610
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6698914, 21.6699066
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6589699, 15.6575050
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2907562, 26.2888031
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6656342, 21.6651573
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0534821, 21.0533409
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8249969, 27.8260651
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4646378, 24.4644547
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2881050, 23.2888947
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4054184, 25.4065094
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1802063, 36.1814423
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8217926, 32.8234711
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5125198, 29.5157013
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0441818, 25.0440216
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7937851, 25.7937012
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9029579, 18.9022522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1559

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 782

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6913456, upper bound: 19.6857328
time: 36.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6927754, upper bound: 19.6843040
time: 43.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7185211, 24.7209587
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0126419, 16.0138779
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5130424, 15.5152740
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0839767, 20.0860596
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0165596, 21.0197411
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3820496, 19.3835907
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6191940, 22.6189957
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5059357, 21.5084648
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4261093, 26.4303665
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3728180, 23.3748169
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4831390, 25.4846115
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6674309, 17.6667976
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3604126, 29.3582230
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3632965, 30.3635941
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5311584, 43.5317688
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1067886, 24.1075134
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0009613, 21.0038376
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1932678, 31.1918793
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5979996, 15.5975075
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5357399, 18.5350609
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0071564, 21.0058517
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6841507, 16.6823120
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8083725, 18.8069763
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7334785, 21.7321587
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0520248, 21.0494766
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5087280, 30.5084229
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6900177, 20.6892281
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6637573, 21.6623116
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6496201, 15.6473694
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2866516, 26.2846222
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6641541, 21.6632042
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0564423, 21.0567741
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8304062, 27.8306808
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4662247, 24.4661751
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2877693, 23.2874603
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4053268, 25.4046631
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1851730, 36.1855850
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8187943, 32.8173294
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5194244, 29.5219345
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0610199, 25.0643463
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7986145, 25.7995224
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9022942, 18.9027977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1019

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1567

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6948682, upper bound: 19.6723833
time: 36.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6947307, upper bound: 19.6725196
time: 40.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7195435, 24.7199326
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0132446, 16.0132751
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5144157, 15.5139046
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0853119, 20.0847244
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0184822, 21.0178185
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3829041, 19.3827362
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6194229, 22.6187668
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5077972, 21.5066032
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4288406, 26.4276352
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3739700, 23.3736649
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4840927, 25.4836578
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6674004, 17.6668243
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3591614, 29.3594742
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3633575, 30.3635330
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5313568, 43.5315857
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1068115, 24.1074944
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0032425, 21.0015564
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1921844, 31.1929550
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5976410, 15.5978642
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5353203, 18.5354881
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0062256, 21.0067787
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6825180, 16.6839466
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8072281, 18.8081169
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7325020, 21.7331352
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0499954, 21.0515137
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5085144, 30.5086365
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6896057, 20.6896400
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6627884, 21.6632767
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6479073, 15.6490784
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2855453, 26.2857285
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6635590, 21.6637993
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0569611, 21.0562515
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8306427, 27.8304520
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4662628, 24.4661255
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2875557, 23.2876740
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4050217, 25.4049683
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1854782, 36.1852875
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8175125, 32.8186340
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5215759, 29.5197830
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0641632, 25.0612106
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7995529, 25.7985840
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9027748, 18.9022598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1524

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6641822, upper bound: 19.6768870
time: 26.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6641822, upper bound: 19.6768870
time: 26.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7201691, 24.7241707
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0160599, 16.0191803
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5217972, 15.5250168
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0906067, 20.0930214
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0299301, 21.0332718
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3871002, 19.3903999
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6161423, 22.6171455
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5150452, 21.5187683
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4377060, 26.4455338
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3711700, 23.3734741
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4819260, 25.4859772
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6654663, 17.6656151
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3718719, 29.3689041
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3689957, 30.3637924
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5290070, 43.5310211
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1076965, 24.1084518
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0012360, 21.0070877
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1968842, 31.1970139
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5995445, 15.5992451
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5363617, 18.5376282
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0097504, 21.0100327
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6978989, 16.6945477
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8126183, 18.8110352
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7366791, 21.7352829
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0698700, 21.0668411
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5106888, 30.5122147
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6950989, 20.6952362
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6689987, 21.6690865
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6603584, 15.6592560
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2876968, 26.2875595
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6663437, 21.6644554
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0558777, 21.0526962
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8293533, 27.8223419
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4671326, 24.4633484
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2922745, 23.2872772
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4068069, 25.4007416
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1824646, 36.1789856
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8212814, 32.8153305
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5175476, 29.5100784
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0409393, 25.0417786
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7940369, 25.7934952
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9019508, 18.9022675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1410

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6919677, upper bound: 19.6937869
time: 36.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6935560, upper bound: 19.6921980
time: 31.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7199707, 24.7243767
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0157928, 16.0194435
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5217896, 15.5250320
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0904694, 20.0931549
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0296249, 21.0335808
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3868713, 19.3906326
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6164627, 22.6168327
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5148315, 21.5189819
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4373093, 26.4459381
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3705444, 23.3740921
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4825974, 25.4853020
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6656265, 17.6654549
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3726196, 29.3681488
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3688507, 30.3639297
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5286713, 43.5313568
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1074524, 24.1086922
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0018158, 21.0065079
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1970215, 31.1968842
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5995712, 15.5992184
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5364380, 18.5375519
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0098648, 21.0099182
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6977921, 16.6946564
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8123894, 18.8112602
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7370300, 21.7349319
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0694885, 21.0672226
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5107040, 30.5122070
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6952972, 20.6950493
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6690903, 21.6689949
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6605186, 15.6590958
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2881012, 26.2871552
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6666794, 21.6641235
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0559769, 21.0526009
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8293533, 27.8223419
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4673233, 24.4631538
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2922363, 23.2873116
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4062424, 25.4012985
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1823883, 36.1790466
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8201370, 32.8164825
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5173111, 29.5103149
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0414047, 25.0413132
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7940445, 25.7934875
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9018822, 18.9023361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1425

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1333

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6925470, upper bound: 19.6919562
time: 40.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6950223, upper bound: 19.6894832
time: 38.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7243805, 24.7196045
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0174599, 16.0146065
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5249939, 15.5222740
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.1055603, 20.1041908
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0229149, 21.0181236
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3832550, 19.3813782
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.5721359, 22.5767670
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5224991, 21.5197372
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4524155, 26.4445648
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3677444, 23.3629913
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4880219, 25.4851227
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6640549, 17.6660004
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3639603, 29.3654480
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3720245, 30.3728256
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5322113, 43.5302124
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1029358, 24.1006927
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0175629, 21.0136566
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2082977, 31.2078400
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6068153, 15.6055946
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5309715, 18.5316200
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0132141, 21.0129585
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6864319, 16.6866741
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8143539, 18.8151360
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7392120, 21.7406197
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0639687, 21.0645218
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5142822, 30.5133972
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6842651, 20.6857224
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6796417, 21.6788979
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6531296, 15.6538429
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2900848, 26.2916031
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6700249, 21.6702003
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0300293, 21.0349579
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8222351, 27.8263245
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4412384, 24.4453430
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2891121, 23.2924843
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3878784, 25.3939209
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1836624, 36.1865997
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7682037, 32.7764740
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5045853, 29.5100555
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0373840, 25.0381012
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7815323, 25.7837448
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8988228, 18.9040794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 825

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1605

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6827314, upper bound: 19.6929088
time: 30.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6827314, upper bound: 19.6929088
time: 31.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7237244, 24.7202568
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0173073, 16.0147591
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5248566, 15.5224037
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.1063004, 20.1034470
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0216255, 21.0194130
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3840637, 19.3805656
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.5760651, 22.5728378
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5226364, 21.5195999
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4512177, 26.4457626
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3656845, 23.3650513
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4878922, 25.4852524
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6652603, 17.6647949
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3639755, 29.3654175
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3725433, 30.3722992
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5321503, 43.5302582
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1023788, 24.1012497
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0158844, 21.0153351
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2076569, 31.2084885
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6055412, 15.6068687
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5314445, 18.5311470
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0126648, 21.0135078
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6862526, 16.6868515
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8143921, 18.8150978
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7394638, 21.7403755
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0640068, 21.0644798
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5135651, 30.5141068
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6852570, 20.6847343
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6785049, 21.6800346
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6534576, 15.6535110
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2903290, 26.2913589
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6696892, 21.6705360
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0324783, 21.0325089
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8222351, 27.8263168
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4428635, 24.4437141
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2894478, 23.2921448
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3899841, 25.3918152
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1846695, 36.1855774
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7731323, 32.7715454
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5055847, 29.5090485
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0381165, 25.0373688
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7830429, 25.7822418
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9038582, 18.8990402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1545

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6790675, upper bound: 19.6891471
time: 31.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6790675, upper bound: 19.6891471
time: 31.10 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 64.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6904109, upper bound: 19.6890287
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6902443, upper bound: 19.6891955
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6913456, upper bound: 19.6857328
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6927754, upper bound: 19.6843040
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6948682, upper bound: 19.6723833
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6947307, upper bound: 19.6725196
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6641822, upper bound: 19.6768870
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6641822, upper bound: 19.6768870
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6919677, upper bound: 19.6937869
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6935560, upper bound: 19.6921980
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6925470, upper bound: 19.6919562
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6950223, upper bound: 19.6894832
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6827314, upper bound: 19.6929088
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6827314, upper bound: 19.6929088
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6790675, upper bound: 19.6891471
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 64.35
Output dim: 26, lower bound: -19.6790675, upper bound: 19.6891471

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7195435, 24.7147217
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0161018, 16.0135269
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5217438, 15.5200348
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0933838, 20.0919838
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0330009, 21.0306892
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3869095, 19.3861122
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6129456, 22.6147804
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5180588, 21.5154343
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4410934, 26.4391785
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3814011, 23.3776016
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4863434, 25.4878998
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6673584, 17.6676407
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3604889, 29.3610001
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3675156, 30.3606491
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5330658, 43.5312958
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1055298, 24.1055756
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0156555, 21.0139427
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1979218, 31.1998367
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5990906, 15.5995712
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5367737, 18.5370598
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0120277, 21.0121117
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6920891, 16.6939640
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8150558, 18.8158302
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7366562, 21.7384529
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0638428, 21.0651207
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5103912, 30.5111618
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6898422, 20.6908150
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6700287, 21.6719971
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6576462, 15.6604652
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2885666, 26.2921906
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6654434, 21.6662407
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0507050, 21.0501633
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8231125, 27.8209763
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4604492, 24.4624367
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2866783, 23.2859993
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4067688, 25.4031677
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1811981, 36.1789322
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8265533, 32.8212051
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5174103, 29.5096970
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0429153, 25.0424347
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7925797, 25.7923279
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9020233, 18.9012451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 911

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6895842, upper bound: 19.6830212
time: 34.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6844040, upper bound: 19.6882022
time: 37.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7188110, 24.7154541
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0158043, 16.0138245
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5212936, 15.5204849
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0932846, 20.0920753
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0324898, 21.0312004
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3863678, 19.3866501
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6134338, 22.6142883
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5177078, 21.5157776
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4403534, 26.4399185
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3813019, 23.3777008
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4863892, 25.4878616
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6674271, 17.6675758
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3612213, 29.3602600
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3674164, 30.3607483
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5330048, 43.5313568
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1050339, 24.1060677
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0156250, 21.0139771
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1978302, 31.1999207
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5990982, 15.5995598
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5367889, 18.5370445
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0122643, 21.0118752
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6921349, 16.6939182
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8150024, 18.8158875
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7365952, 21.7385178
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0638580, 21.0650978
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5101929, 30.5113525
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6895294, 20.6911278
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6700592, 21.6719704
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6576805, 15.6604347
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2888947, 26.2918701
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6654434, 21.6662369
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0511246, 21.0497360
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8236160, 27.8204727
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4609528, 24.4619370
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2873878, 23.2852859
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4067154, 25.4032288
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1811523, 36.1789627
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8265076, 32.8212585
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5173035, 29.5097961
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0429153, 25.0424347
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7926865, 25.7922211
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9020004, 18.9012642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 782

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6881387, upper bound: 19.6885184
time: 41.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6895675, upper bound: 19.6870886
time: 43.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7211533, 24.7240181
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0152435, 16.0163956
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5224228, 15.5235558
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0925064, 20.0928040
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0303001, 21.0311050
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3878937, 19.3876801
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6164398, 22.6150436
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5182877, 21.5185547
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4448318, 26.4449310
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3778610, 23.3789749
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4870148, 25.4866104
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6672668, 17.6667137
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3652954, 29.3647156
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3617249, 30.3654404
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5320282, 43.5323639
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1069946, 24.1075554
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0134506, 21.0141449
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1989594, 31.1983032
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5993938, 15.5995331
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5373230, 18.5365906
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0120125, 21.0118904
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6929855, 16.6923943
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8155632, 18.8156929
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7382278, 21.7381210
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0643158, 21.0637131
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5108032, 30.5104370
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6913528, 20.6902924
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6699600, 21.6698685
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6589966, 15.6574631
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2908707, 26.2885818
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6655121, 21.6650772
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0532608, 21.0534058
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8242798, 27.8261108
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4644394, 24.4643402
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2876091, 23.2888451
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4048309, 25.4066544
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1798248, 36.1816406
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8211212, 32.8236542
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5115356, 29.5158768
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0441513, 25.0440216
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7936707, 25.7937469
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9028854, 18.9022865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 888

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1420

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6911789, upper bound: 19.6840457
time: 37.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6896577, upper bound: 19.6855661
time: 43.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7211990, 24.7239914
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0151596, 16.0165367
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5222626, 15.5236130
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0925293, 20.0928726
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0300865, 21.0313644
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3877563, 19.3878670
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6164246, 22.6150665
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5180817, 21.5188217
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4441910, 26.4454956
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3778534, 23.3789444
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4866943, 25.4868469
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6671600, 17.6667175
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3654327, 29.3643875
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3619995, 30.3645020
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5319672, 43.5324249
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1069565, 24.1075897
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0131836, 21.0143929
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1989136, 31.1982880
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5993786, 15.5995369
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5371246, 18.5367508
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0120888, 21.0118599
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6930809, 16.6923580
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8157234, 18.8155899
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7382126, 21.7381172
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0643387, 21.0635948
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5106506, 30.5104828
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6912994, 20.6903610
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6698532, 21.6699066
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6589279, 15.6575050
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2905350, 26.2888031
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6656342, 21.6650429
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0534821, 21.0531235
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8249969, 27.8253403
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4646378, 24.4642563
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2881050, 23.2884026
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4054184, 25.4059219
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1802063, 36.1810608
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8217926, 32.8227997
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5125198, 29.5147247
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0441742, 25.0440216
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7937851, 25.7935867
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9029579, 18.9021873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1365

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6825414, upper bound: 19.6827962
time: 33.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6912670, upper bound: 19.6740636
time: 40.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7181168, 24.7194023
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0114365, 16.0130653
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5112572, 15.5150223
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0839462, 20.0858574
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0144157, 21.0182037
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3771820, 19.3804245
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6172867, 22.6168633
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5031738, 21.5062981
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4141693, 26.4227905
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3750916, 23.3742294
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4759293, 25.4800529
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6669350, 17.6687241
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3540497, 29.3475571
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3591232, 30.3470383
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5310211, 43.5317230
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1047668, 24.1061134
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0005569, 21.0045624
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1924286, 31.1950226
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5976906, 15.5985374
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5353279, 18.5366745
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0070953, 21.0058632
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6840286, 16.6821060
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8080521, 18.8075600
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7329597, 21.7340622
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0520020, 21.0494461
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5078506, 30.5117798
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6870728, 20.6895142
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6631622, 21.6644440
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6464005, 15.6462440
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2822647, 26.2835922
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6640778, 21.6631927
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0521431, 21.0468750
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8093719, 27.7977829
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4581070, 24.4533386
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2746048, 23.2666206
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3911972, 25.3822937
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1743164, 36.1683807
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8036194, 32.7932892
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4905243, 29.4772568
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0559616, 25.0575027
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7944717, 25.7931519
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9014244, 18.9007111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1367

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1507

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6943977, upper bound: 19.6683463
time: 38.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6880850, upper bound: 19.6718173
time: 78.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7169724, 24.7205467
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0118332, 16.0126724
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5127983, 15.5134926
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0837708, 20.0860252
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0150185, 21.0176048
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3788757, 19.3787270
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6170731, 22.6170731
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5037689, 21.5056877
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4185333, 26.4184265
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3722382, 23.3770828
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4785843, 25.4773979
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6693535, 17.6663055
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3497314, 29.3518677
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3467484, 30.3594055
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5310974, 43.5316467
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1053848, 24.1054916
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0016785, 21.0034332
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1963959, 31.1910477
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5990181, 15.5972099
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5373573, 18.5346451
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0071640, 21.0057983
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6839447, 16.6821899
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8089447, 18.8066635
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7353859, 21.7316437
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0519867, 21.0494499
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5120926, 30.5075455
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6902924, 20.6862946
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6658783, 21.6617279
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6485023, 15.6441422
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2856064, 26.2802505
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6641541, 21.6631317
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0465431, 21.0524826
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7975082, 27.8096466
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4533844, 24.4580536
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2669296, 23.2742920
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3829651, 25.3905258
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1679840, 36.1747131
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7947693, 32.8021393
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4747314, 29.4930496
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0541763, 25.0592880
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7922592, 25.7953720
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9002037, 18.9019318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1427

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1298

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6945770, upper bound: 19.6724278
time: 43.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6946382, upper bound: 19.6723665
time: 29.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7194366, 24.7233429
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0160370, 16.0191536
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5212631, 15.5244064
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0887527, 20.0909195
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0295792, 21.0328522
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3848495, 19.3878822
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6167068, 22.6175537
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5145187, 21.5181656
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4360886, 26.4437027
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3713379, 23.3735733
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4811783, 25.4851303
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6647148, 17.6649628
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3709259, 29.3680649
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3689804, 30.3637848
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5290222, 43.5310211
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1069641, 24.1076393
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0014648, 21.0072441
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1963806, 31.1965790
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5993233, 15.5991287
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5362854, 18.5375633
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0095253, 21.0098610
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6975861, 16.6942787
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8123207, 18.8107643
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7360077, 21.7346840
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0698128, 21.0667953
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5105820, 30.5121307
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6948929, 20.6950531
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6679230, 21.6681671
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6599541, 15.6589127
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2869492, 26.2868958
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6661339, 21.6642723
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0559692, 21.0528030
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8293915, 27.8223801
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4674454, 24.4636383
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2919998, 23.2870560
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4068375, 25.4007797
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1818085, 36.1783524
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8220215, 32.8159943
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5175781, 29.5101166
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0406342, 25.0414505
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7939682, 25.7934036
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9022141, 18.9023361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1329

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1561

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6903583, upper bound: 19.6751584
time: 35.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6733486, upper bound: 19.6921780
time: 32.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7193451, 24.7234230
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0160370, 16.0191574
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5211792, 15.5244865
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0885010, 20.0911713
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0295181, 21.0329208
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3845825, 19.3881531
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6165543, 22.6177063
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5144424, 21.5182495
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4358749, 26.4439163
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3712692, 23.3736420
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4810715, 25.4852295
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6648140, 17.6648598
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3710327, 29.3679581
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3689804, 30.3637848
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5290222, 43.5310364
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1068726, 24.1077271
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0013885, 21.0073204
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1964417, 31.1965103
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5994263, 15.5990238
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5363007, 18.5375557
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0095863, 21.0098000
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6976318, 16.6942329
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8123512, 18.8107414
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7360764, 21.7346077
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0698204, 21.0667839
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5106125, 30.5121002
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6949158, 20.6950264
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6680756, 21.6680145
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6600189, 15.6588516
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2870331, 26.2868118
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6661644, 21.6642418
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0559845, 21.0527878
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8293915, 27.8223801
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4674225, 24.4636612
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2920456, 23.2870026
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4068451, 25.4007721
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1818390, 36.1783295
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8219604, 32.8160553
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5175858, 29.5101089
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0406113, 25.0414810
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7939377, 25.7934265
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9020157, 18.9025307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1301

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6914889, upper bound: 19.6921338
time: 33.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6934918, upper bound: 19.6901371
time: 37.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7159500, 24.7196007
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0109901, 16.0141487
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5192490, 15.5220070
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0865250, 20.0885735
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0328445, 21.0362206
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3859100, 19.3892479
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6128922, 22.6137314
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5100861, 21.5134888
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4298859, 26.4376526
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3699799, 23.3733521
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4843445, 25.4874802
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6627274, 17.6631584
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3690109, 29.3650742
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3671494, 30.3619614
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5257721, 43.5281525
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1081848, 24.1092949
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0014877, 21.0061264
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1941147, 31.1942215
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6001205, 15.5999088
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5362930, 18.5374718
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0093727, 21.0095673
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6943398, 16.6918392
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8124008, 18.8112755
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7369537, 21.7348709
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0690918, 21.0673180
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5107117, 30.5122147
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6941414, 20.6940041
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6711502, 21.6714363
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6590843, 15.6580353
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2870941, 26.2863388
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6639099, 21.6617508
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0546379, 21.0513916
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8267365, 27.8195343
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4674301, 24.4633484
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2924194, 23.2874603
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4062271, 25.4012604
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1820831, 36.1787109
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8205795, 32.8162994
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5180969, 29.5108871
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0409927, 25.0409393
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7940979, 25.7935791
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9018593, 18.9023247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1396

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1576

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6916028, upper bound: 19.6729654
time: 39.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6735468, upper bound: 19.6910182
time: 34.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7151947, 24.7203560
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0105019, 16.0146446
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5187607, 15.5224953
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0858917, 20.0892105
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0322647, 21.0368042
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3854828, 19.3896751
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6133575, 22.6132736
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5093384, 21.5142365
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4290161, 26.4385223
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3698120, 23.3735275
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4847794, 25.4870453
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6633301, 17.6625595
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3695450, 29.3645325
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3668747, 30.3622360
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5254517, 43.5284729
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1080475, 24.1094284
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0014420, 21.0061722
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1943588, 31.1939774
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6002617, 15.5997696
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5363541, 18.5374107
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0095100, 21.0094337
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6949730, 16.6912041
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8124084, 18.8112755
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7369690, 21.7348480
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0695801, 21.0668297
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5107117, 30.5122147
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6942482, 20.6938934
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6715317, 21.6710587
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6594582, 15.6576614
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2872849, 26.2861557
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6643066, 21.6613541
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0547600, 21.0512657
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8265381, 27.8197250
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4675217, 24.4632568
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2923889, 23.2874908
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4062042, 25.4012833
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1820526, 36.1787338
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8199387, 32.8169327
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5178833, 29.5111008
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0410385, 25.0409012
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7941360, 25.7935410
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9018669, 18.9023170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 824

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 963

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6683722, upper bound: 19.6892301
time: 33.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6947699, upper bound: 19.6633398
time: 31.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7319946, 24.7184181
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0211487, 16.0141983
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5301857, 15.5214958
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.1086121, 20.1037331
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0315628, 21.0169945
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3881226, 19.3806610
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.5711441, 22.5833626
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5272064, 21.5190277
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4592056, 26.4436569
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3705521, 23.3626251
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4878464, 25.4856796
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6636734, 17.6700020
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3628616, 29.3727264
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3780746, 30.3722076
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5355530, 43.5297089
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1066437, 24.1002960
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0175247, 21.0137596
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2082672, 31.2079468
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6076088, 15.6054554
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5309563, 18.5316849
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0131302, 21.0133286
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6866341, 16.6866188
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8143005, 18.8169289
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7391129, 21.7406006
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0639687, 21.0645218
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5141983, 30.5133896
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6842346, 20.6858368
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6796036, 21.6821327
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6528816, 15.6560020
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2893372, 26.2975998
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6704865, 21.6700897
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0294647, 21.0390129
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8218765, 27.8275299
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4402237, 24.4520760
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2888985, 23.2943153
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3885345, 25.3938599
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1832047, 36.1896210
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7714996, 32.7762985
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5096436, 29.5098190
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0366821, 25.0424423
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7808838, 25.7883911
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8980370, 18.9098701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1427

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 824

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6159857, upper bound: 19.6535713
time: 32.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6434213, upper bound: 19.6261492
time: 32.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7231903, 24.7196045
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0170593, 16.0146065
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5242195, 15.5222740
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.1051025, 20.1041908
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0217896, 21.0181236
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3825378, 19.3813782
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.5721359, 22.5757675
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5217972, 21.5197372
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4515076, 26.4445648
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3673859, 23.3629913
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4880219, 25.4849701
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6640549, 17.6656151
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3639603, 29.3643494
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3713913, 30.3728256
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5317078, 43.5302124
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1025391, 24.1006927
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0175629, 21.0136147
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2082977, 31.2078018
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6066742, 15.6055946
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5309715, 18.5316086
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0132141, 21.0128822
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6863747, 16.6866741
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8143539, 18.8150940
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7391968, 21.7406197
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0639687, 21.0645180
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5142746, 30.5133972
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6842651, 20.6856995
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6796417, 21.6788597
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6531296, 15.6535950
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2900848, 26.2908554
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6699219, 21.6702003
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0300293, 21.0343933
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8222351, 27.8259735
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4412384, 24.4443283
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2891121, 23.2922668
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3878174, 25.3939209
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1836624, 36.1861572
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7680206, 32.7764740
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5043411, 29.5100555
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0373840, 25.0374069
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7815323, 25.7830963
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8988228, 18.9032860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1406

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6757200, upper bound: 19.6914094
time: 44.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6812271, upper bound: 19.6859042
time: 35.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7247391, 24.7202110
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0172081, 16.0149460
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5246124, 15.5232849
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.1060944, 20.1042404
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0209618, 21.0217934
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3836823, 19.3819351
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.5771790, 22.5724068
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5222473, 21.5210114
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4508972, 26.4467926
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3668213, 23.3650360
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4878769, 25.4852524
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6651382, 17.6653214
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3654938, 29.3649979
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3723755, 30.3728867
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5317993, 43.5315704
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1023865, 24.1012573
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0171051, 21.0153198
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2076263, 31.2086182
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6053658, 15.6075745
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5312309, 18.5320816
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0124588, 21.0143394
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6861572, 16.6876297
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8156853, 18.8150063
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7393417, 21.7412071
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0639458, 21.0645790
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5135574, 30.5148697
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6852264, 20.6846581
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6791000, 21.6800308
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6534424, 15.6535416
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2901993, 26.2917633
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6694832, 21.6717796
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0339050, 21.0321312
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8251190, 27.8255157
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4447098, 24.4432030
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2897415, 23.2919922
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3898010, 25.3917542
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1885529, 36.1845016
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7731323, 32.7719040
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5054779, 29.5088959
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0415497, 25.0364227
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7855911, 25.7815247
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9073067, 18.8980865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1669

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1318

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6789106, upper bound: 19.6888684
time: 34.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6787881, upper bound: 19.6889895
time: 29.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7236862, 24.7202568
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0173073, 16.0146561
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5248566, 15.5221481
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.1063004, 20.1032486
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0216255, 21.0187569
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3840637, 19.3801804
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.5756378, 22.5728378
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5226364, 21.5192032
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4512177, 26.4454346
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3656693, 23.3650513
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4878922, 25.4852600
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6652603, 17.6646805
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3635559, 29.3654175
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3725433, 30.3721390
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5321503, 43.5299225
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1023788, 24.1012497
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0158768, 21.0153351
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2076569, 31.2084732
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6055412, 15.6066952
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5314445, 18.5309334
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0126648, 21.0133057
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6862526, 16.6867542
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8142967, 18.8150978
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7394638, 21.7402496
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0640068, 21.0644188
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5135651, 30.5140839
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6852570, 20.6847115
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6784897, 21.6800346
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6534576, 15.6534958
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2903290, 26.2912292
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6696892, 21.6703262
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0321045, 21.0325089
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8214264, 27.8263168
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4423599, 24.4437141
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2892990, 23.2921448
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3899307, 25.3918152
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1836090, 36.1855774
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7731323, 32.7715378
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5054398, 29.5090485
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0371780, 25.0373688
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7823486, 25.7822418
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9029045, 18.8990402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1458

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6777871, upper bound: 19.6877973
time: 31.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6777809, upper bound: 19.6878596
time: 38.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 72.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6895842, upper bound: 19.6830212
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6844040, upper bound: 19.6882022
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6881387, upper bound: 19.6885184
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6895675, upper bound: 19.6870886
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6911789, upper bound: 19.6840457
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6896577, upper bound: 19.6855661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6825414, upper bound: 19.6827962
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6912670, upper bound: 19.6740636
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6943977, upper bound: 19.6683463
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6880850, upper bound: 19.6718173
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6945770, upper bound: 19.6724278
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6946382, upper bound: 19.6723665
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6903583, upper bound: 19.6751584
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6733486, upper bound: 19.6921780
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6914889, upper bound: 19.6921338
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6934918, upper bound: 19.6901371
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6916028, upper bound: 19.6729654
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6735468, upper bound: 19.6910182
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6683722, upper bound: 19.6892301
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6947699, upper bound: 19.6633398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6159857, upper bound: 19.6535713
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6434213, upper bound: 19.6261492
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6757200, upper bound: 19.6914094
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6812271, upper bound: 19.6859042
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6789106, upper bound: 19.6888684
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6787881, upper bound: 19.6889895
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6777871, upper bound: 19.6877973
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 26, lower bound: -19.6777809, upper bound: 19.6878596

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7115479, 24.7107315
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0052261, 16.0048485
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5163155, 15.5164261
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0776367, 20.0788918
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0193405, 21.0189056
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3741913, 19.3755264
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6130905, 22.6149406
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5030136, 21.5027046
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4139557, 26.4166031
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3641434, 23.3629913
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4743271, 25.4777756
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6638451, 17.6635704
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3519363, 29.3509903
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3676071, 30.3606949
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5348663, 43.5343018
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.0987091, 24.0997467
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0104980, 21.0115166
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1900101, 31.1903763
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5966778, 15.5960922
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5360374, 18.5362434
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0135155, 21.0128670
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6891365, 16.6886673
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8112183, 18.8114586
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7290268, 21.7288971
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0545273, 21.0538254
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5085831, 30.5093536
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6917114, 20.6914749
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6635284, 21.6641731
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6508102, 15.6517906
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2867813, 26.2886124
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6617279, 21.6603546
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0513496, 21.0507507
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8220863, 27.8196869
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4596024, 24.4615097
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2820091, 23.2804947
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4014664, 25.3970108
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1843796, 36.1821442
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8276520, 32.8223343
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5167313, 29.5088806
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0406952, 25.0420227
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7945328, 25.7947845
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8978271, 18.8991508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6885967, upper bound: 19.6727413
time: 37.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6793058, upper bound: 19.6820337
time: 41.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7155457, 24.7067261
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0074234, 16.0026550
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5181313, 15.5146027
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0802917, 20.0762367
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0212173, 21.0170250
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3763199, 19.3733940
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6131058, 22.6149292
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5053329, 21.5003853
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4185181, 26.4120407
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3667831, 23.3603516
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4762115, 25.4758835
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6632881, 17.6641273
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3504715, 29.3524399
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3675537, 30.3607483
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5360565, 43.5330963
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.0997009, 24.0987587
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0132294, 21.0087776
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1884537, 31.1919250
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5956097, 15.5971603
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5359612, 18.5363235
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0127754, 21.0136032
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6867905, 16.6910095
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8106842, 18.8120003
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7271042, 21.7308121
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0525589, 21.0558090
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5085831, 30.5093689
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6904984, 20.6926880
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6622086, 21.6654930
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6489716, 15.6536293
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2849960, 26.2904053
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6595535, 21.6625252
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0512886, 21.0508080
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8218193, 27.8199463
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4595184, 24.4615860
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2811699, 23.2813339
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4006119, 25.3978653
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1843948, 36.1821136
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8276825, 32.8223190
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5165939, 29.5090179
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0425110, 25.0402145
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7950287, 25.7942810
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8999252, 18.8970490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1510

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1396

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6813328, upper bound: 19.6879285
time: 32.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6841304, upper bound: 19.6851309
time: 292.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7187653, 24.7154350
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0157471, 16.0136871
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5213966, 15.5204315
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0932007, 20.0920067
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0324554, 21.0309525
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3863144, 19.3864555
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6134186, 22.6142578
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5176544, 21.5155106
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4404373, 26.4393692
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3813095, 23.3777313
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4864883, 25.4876366
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6675186, 17.6675644
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3610840, 29.3604355
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3671494, 30.3614197
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5330200, 43.5313110
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1050491, 24.1060333
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0156479, 21.0137405
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1978836, 31.1999207
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5991135, 15.5995560
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5368195, 18.5368767
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0121956, 21.0118408
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6920433, 16.6938629
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8148422, 18.8158302
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7366180, 21.7385139
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0638351, 21.0651894
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5102921, 30.5112915
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6895142, 20.6910629
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6701355, 21.6719360
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6577034, 15.6603889
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2889938, 26.2916336
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6653290, 21.6661568
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0509186, 21.0498123
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8228912, 27.8205261
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4607620, 24.4618340
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2869072, 23.2852478
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4061432, 25.4033813
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1808014, 36.1791763
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8258362, 32.8214493
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5163422, 29.5099869
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0428772, 25.0424271
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7925720, 25.7922668
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9019356, 18.9012985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1378

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6871063, upper bound: 19.6882709
time: 32.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6878912, upper bound: 19.6874855
time: 47.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7188110, 24.7154083
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0156631, 16.0138245
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5212440, 15.5204849
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0932236, 20.0920753
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0322418, 21.0312004
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3861771, 19.3866501
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6134033, 22.6142883
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5174484, 21.5157776
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4398041, 26.4399185
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3813019, 23.3777008
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4861679, 25.4878616
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6674194, 17.6675758
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3612213, 29.3601151
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3674164, 30.3604813
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5329590, 43.5313568
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1050034, 24.1060677
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0153809, 21.0139771
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1978378, 31.1999207
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5990982, 15.5995598
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5366211, 18.5370445
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0122643, 21.0118103
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6921349, 16.6938267
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8150024, 18.8157272
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7365952, 21.7385178
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0638580, 21.0650711
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5101395, 30.5113525
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6894608, 20.6911278
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6700211, 21.6719704
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6576347, 15.6604347
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2886658, 26.2918701
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6654434, 21.6661224
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0511246, 21.0495262
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8236160, 27.8197556
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4609528, 24.4617500
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2873878, 23.2848053
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4067154, 25.4026566
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1811523, 36.1785965
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8265076, 32.8205872
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5173035, 29.5088348
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0429001, 25.0424347
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7926865, 25.7921066
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9020004, 18.9011993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1393

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1330

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6885627, upper bound: 19.6865970
time: 37.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6890780, upper bound: 19.6860840
time: 37.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7204285, 24.7232132
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0141754, 16.0150948
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5202141, 15.5210648
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0930557, 20.0933952
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0291824, 21.0296326
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3879700, 19.3877411
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6129303, 22.6121368
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5167694, 21.5167465
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4422760, 26.4420853
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3757858, 23.3766479
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4865417, 25.4861946
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6680069, 17.6673012
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3634949, 29.3632889
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3636169, 30.3676834
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5307465, 43.5307770
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1073227, 24.1079178
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0102081, 21.0103798
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2018280, 31.2007751
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5986748, 15.5984745
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5369797, 18.5361748
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0099792, 21.0093842
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6927834, 16.6924400
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8154984, 18.8155479
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7380295, 21.7378922
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0655479, 21.0650444
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5127945, 30.5119095
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6915359, 20.6904411
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6694717, 21.6692810
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6576233, 15.6561432
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2920990, 26.2898026
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6653748, 21.6648674
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0502510, 21.0508690
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8188477, 27.8218002
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4582520, 24.4591217
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2851677, 23.2869873
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4023514, 25.4045181
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1787415, 36.1807251
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8182144, 32.8212662
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5086517, 29.5136185
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0420227, 25.0421982
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7922668, 25.7927017
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9001961, 18.9000702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1511

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6908058, upper bound: 19.6761101
time: 27.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6830276, upper bound: 19.6836558
time: 40.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7203445, 24.7232933
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0139313, 16.0153313
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5199242, 15.5213547
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0931015, 20.0933380
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0288239, 21.0299873
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3879623, 19.3877525
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6135254, 22.6115417
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5164719, 21.5170441
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4419785, 26.4423752
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3755264, 23.3769073
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4865952, 25.4861374
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6678543, 17.6674461
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3638611, 29.3629227
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3639526, 30.3673401
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5304260, 43.5310974
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1073685, 24.1078720
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0096893, 21.0108986
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.2014465, 31.2011642
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5983429, 15.5988083
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5369110, 18.5362434
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0095139, 21.0098457
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6930351, 16.6921864
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8154221, 18.8156242
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7379990, 21.7379150
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0656395, 21.0649452
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5122910, 30.5124130
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6914978, 20.6904755
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6693878, 21.6693687
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6576691, 15.6560974
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2920914, 26.2898102
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6653061, 21.6649361
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0507240, 21.0503998
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8199768, 27.8206711
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4592133, 24.4581604
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2857552, 23.2863998
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4027100, 25.4041672
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1789093, 36.1805573
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8187332, 32.8207474
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5092926, 29.5129776
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0423279, 25.0418854
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7926331, 25.7923431
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9006844, 18.8995895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1521

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1346

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6891036, upper bound: 19.6854676
time: 46.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6895593, upper bound: 19.6850109
time: 36.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7199936, 24.7229767
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0094757, 16.0113678
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5227203, 15.5242081
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0942688, 20.0950012
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0300941, 21.0314140
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3908005, 19.3918571
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6222610, 22.6198044
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5164566, 21.5174103
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4378357, 26.4397507
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3630524, 23.3654327
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4807892, 25.4815788
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6669159, 17.6664276
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3691940, 29.3674011
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3658066, 30.3689957
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5320282, 43.5325317
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1081467, 24.1091309
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9968872, 20.9990845
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1916428, 31.1903534
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5983505, 15.5982380
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5370712, 18.5366669
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0100555, 21.0097275
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6842957, 16.6832237
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8189850, 18.8181381
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7300415, 21.7292023
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0654221, 21.0638428
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5095901, 30.5093460
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6876984, 20.6863098
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6660080, 21.6654015
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6578751, 15.6561089
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2911682, 26.2889023
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6547546, 21.6536217
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0537033, 21.0529175
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8292999, 27.8291931
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4674530, 24.4667397
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2873611, 23.2876244
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3950119, 25.3953934
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1862488, 36.1857834
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8004761, 32.8011093
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5046768, 29.5073929
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0581436, 25.0567703
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7979355, 25.7966309
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9067307, 18.9047470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1443

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1554

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6742554, upper bound: 19.6570221
time: 33.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6742554, upper bound: 19.6570221
time: 34.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7169800, 24.7184715
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0102463, 16.0120659
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5086594, 15.5127525
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0823059, 20.0844536
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0107689, 21.0150185
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3774414, 19.3807487
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6189041, 22.6187210
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.4995956, 21.5031738
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4086685, 26.4180908
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3727417, 23.3722763
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4738312, 25.4782715
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6673698, 17.6691322
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3531265, 29.3465118
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3596954, 30.3474808
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5313721, 43.5321808
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1048126, 24.1061745
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9953766, 21.0001144
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1896744, 31.1918030
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5967751, 15.5974693
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5347900, 18.5360527
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0060005, 21.0042610
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6811752, 16.6786537
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8072205, 18.8063507
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7309113, 21.7316895
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0504990, 21.0471878
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5069046, 30.5108795
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6861305, 20.6883659
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6626663, 21.6638336
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6445465, 15.6439209
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2821579, 26.2834625
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6637688, 21.6624565
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0533562, 21.0478973
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8115158, 27.7996750
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4606781, 24.4557190
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2747345, 23.2667236
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3897629, 25.3806152
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1771393, 36.1710358
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8001251, 32.7888947
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4916534, 29.4783478
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0628128, 25.0659409
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7970123, 25.7959213
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9016953, 18.9011993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 842

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6798278, upper bound: 19.6673537
time: 31.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6934014, upper bound: 19.6537867
time: 37.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7171707, 24.7182655
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0104370, 16.0118713
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5089722, 15.5124207
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0825577, 20.0842018
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0112419, 21.0145454
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3775177, 19.3806839
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6191254, 22.6184998
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5000763, 21.5027008
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4094696, 26.4172974
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3731079, 23.3718872
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4741516, 25.4779587
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6673393, 17.6691589
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3530045, 29.3466339
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3595734, 30.3476105
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5314636, 43.5320740
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1048126, 24.1061630
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9960709, 20.9993706
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1892166, 31.1922607
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5966148, 15.5976162
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5347061, 18.5361404
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0054893, 21.0047722
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6805878, 16.6792450
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8068542, 18.8067284
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7305908, 21.7320099
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0497437, 21.0479431
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5069504, 30.5108337
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6859245, 20.6885681
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6625519, 21.6639557
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6440735, 15.6443939
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2821350, 26.2834854
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6633415, 21.6628723
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0531731, 21.0480843
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8112564, 27.7999268
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4604721, 24.4559059
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2747116, 23.2667465
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3895111, 25.3808670
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1767883, 36.1712189
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7992249, 32.7898102
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4916077, 29.4784088
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0643997, 25.0643539
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7972336, 25.7957001
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9019089, 18.9009857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1521

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1486

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6841752, upper bound: 19.6612229
time: 32.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6775193, upper bound: 19.6678794
time: 40.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7165375, 24.7200203
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0118027, 16.0126419
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5126419, 15.5133095
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0836563, 20.0858994
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0151825, 21.0176430
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3795776, 19.3792992
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6161041, 22.6162186
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5040741, 21.5059204
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4181137, 26.4179535
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3713837, 23.3761368
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4759369, 25.4750481
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6694031, 17.6663666
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3500671, 29.3522949
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3460846, 30.3586655
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5310669, 43.5316010
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1048889, 24.1049385
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0000305, 21.0015984
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1955414, 31.1902924
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5990219, 15.5972347
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5378265, 18.5351715
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0075874, 21.0062332
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6839218, 16.6821671
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8084641, 18.8062706
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7351913, 21.7314911
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0525093, 21.0500755
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5120010, 30.5074921
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6916389, 20.6878052
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6653900, 21.6613007
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6484604, 15.6441193
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2845306, 26.2793274
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6639023, 21.6629486
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0463028, 21.0522079
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7981491, 27.8100967
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4549713, 24.4598694
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2680740, 23.2753029
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3819733, 25.3893509
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1685944, 36.1752777
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7925491, 32.7995224
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4746552, 29.4929581
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0542831, 25.0593643
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7923431, 25.7954483
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8994598, 18.9011002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1428

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1700

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6918331, upper bound: 19.6677907
time: 40.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6899287, upper bound: 19.6696846
time: 38.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7164536, 24.7201080
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0118027, 16.0126457
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5126114, 15.5133362
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0836411, 20.0859146
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0150604, 21.0177650
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3794403, 19.3794365
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6162186, 22.6161118
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5040207, 21.5059738
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4180450, 26.4180145
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3712845, 23.3762360
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4762344, 25.4747543
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6694031, 17.6663589
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3501587, 29.3522034
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3460007, 30.3587570
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5310516, 43.5316162
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1048431, 24.1049881
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9998474, 21.0017891
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1956635, 31.1901703
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5990410, 15.5972137
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5378799, 18.5351181
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0075951, 21.0062256
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6839256, 16.6821632
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8085556, 18.8061829
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7352448, 21.7314415
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0526237, 21.0499573
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5120316, 30.5074615
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6918068, 20.6876373
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6654434, 21.6612473
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6484680, 15.6441116
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2846832, 26.2791748
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6639557, 21.6628914
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0462723, 21.0522385
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.7979584, 27.8102875
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4552155, 24.4596252
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2679443, 23.2754326
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.3817978, 25.3895340
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1685333, 36.1753387
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.7921524, 32.7999268
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.4746399, 29.4929733
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0542526, 25.0593948
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7923279, 25.7954712
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8993683, 18.9011993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1533

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1490

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6938280, upper bound: 19.6687040
time: 38.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6881754, upper bound: 19.6714535
time: 37.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7132416, 24.7083549
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0163841, 16.0188713
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5203094, 15.5237083
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0814667, 20.0867920
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0173569, 21.0262260
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3742828, 19.3821487
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6166763, 22.6175156
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5078735, 21.5145569
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4361038, 26.4436569
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3746796, 23.3725204
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4801865, 25.4832344
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6627998, 17.6644135
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3706207, 29.3676834
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3577652, 30.3577118
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5244293, 43.5311737
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1046143, 24.1030159
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0038605, 21.0058289
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1963806, 31.1965790
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5974388, 15.5989761
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5303497, 18.5359306
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0003624, 21.0058479
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6968651, 16.6973839
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8086548, 18.8011436
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7358093, 21.7351036
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0665016, 21.0657349
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5104370, 30.5123901
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6936607, 20.6937332
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6676178, 21.6653748
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6598244, 15.6592941
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2829590, 26.2860413
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6618690, 21.6655731
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0551834, 21.0513535
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8217087, 27.8075180
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4620285, 24.4536514
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2913895, 23.2865067
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4050140, 25.4009323
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1627808, 36.1432648
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8193665, 32.8162460
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5172424, 29.5098419
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0253372, 25.0132294
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7834396, 25.7736664
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8902092, 18.8792381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1371

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1307

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6893628, upper bound: 19.6740004
time: 43.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6891843, upper bound: 19.6741797
time: 40.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7044373, 24.7171593
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0157509, 16.0195084
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5205688, 15.5234566
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0846252, 20.0836296
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0229568, 21.0206337
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3791199, 19.3773079
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6166687, 22.6175232
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5109177, 21.5115204
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4360352, 26.4437256
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3702850, 23.3769150
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4792862, 25.4841385
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6641655, 17.6630516
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3705292, 29.3677673
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3629074, 30.3525696
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5291901, 43.5264130
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1023407, 24.1052895
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0000458, 21.0096397
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1963806, 31.1965637
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5991707, 15.5972481
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5346527, 18.5316238
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0055122, 21.0006981
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.7006874, 16.6935577
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8027039, 18.8070869
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7364197, 21.7344856
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0687447, 21.0634804
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5108337, 30.5119934
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6935692, 20.6938248
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6651306, 21.6678619
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6603355, 15.6587830
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2860947, 26.2829056
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6674309, 21.6600075
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0545273, 21.0520134
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8145294, 27.8146896
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4574661, 24.4582214
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2914505, 23.2864494
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4069901, 25.3989563
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1467133, 36.1593399
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8222656, 32.8133469
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5173035, 29.5097809
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0124130, 25.0261536
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7742310, 25.7828827
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.8791161, 18.8903275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1347

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1765

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6727553, upper bound: 19.6828287
time: 32.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6639868, upper bound: 19.6915926
time: 41.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7164764, 24.7200851
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0154839, 16.0185509
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5214882, 15.5247383
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0903320, 20.0928726
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0264282, 21.0299797
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3845673, 19.3881416
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6166992, 22.6178017
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5144196, 21.5182266
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4346466, 26.4426422
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3708878, 23.3732147
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4808273, 25.4850159
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6642303, 17.6645737
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3704376, 29.3673553
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3641739, 30.3583984
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5285950, 43.5306854
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1069717, 24.1078186
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -21.0000839, 21.0056992
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1971741, 31.1973801
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.5999813, 15.5997849
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5340233, 18.5355568
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0084686, 21.0087929
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6958122, 16.6926556
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8108978, 18.8091507
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7360001, 21.7345695
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0698242, 21.0667953
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5094528, 30.5111465
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6945496, 20.6946983
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6677017, 21.6679001
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6591492, 15.6581345
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2846298, 26.2846603
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6648560, 21.6631432
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0547447, 21.0512505
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8257828, 27.8182983
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4620895, 24.4581833
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2900581, 23.2847061
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4011917, 25.3945312
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1763611, 36.1721115
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8125000, 32.8054581
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5106964, 29.5025024
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0410233, 25.0417633
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7906723, 25.7897339
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9020920, 18.9020157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1371

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1308

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6912250, upper bound: 19.6918768
time: 35.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6912329, upper bound: 19.6918684
time: 32.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -33.9308853, -0.5984228, -33.9308853, -0.5984228, -24.7160110, 24.7205544
1: -13.1288700, 7.4427280, -13.1288700, 7.4427280, -16.0154305, 16.0186043
2: -12.1680202, 8.6329422, -12.1680202, 8.6329422, -15.5214424, 15.5247955
3: -26.9002285, -2.2060075, -26.9002285, -2.2060075, -20.0902023, 20.0930061
4: -16.8726807, 11.0384665, -16.8726807, 11.0384665, -21.0265732, 21.0298347
5: -21.6388016, 3.2740502, -21.6388016, 3.2740502, -19.3845673, 19.3881416
6: -34.7174721, -7.5418568, -34.7174721, -7.5418568, -22.6166458, 22.6178551
7: -20.9355278, 6.2501950, -20.9355278, 6.2501950, -21.5144196, 21.5182266
8: -31.0415363, 5.0150971, -31.0415363, 5.0150971, -26.4346008, 26.4426880
9: -18.9931679, 8.0394125, -18.9931679, 8.0394125, -23.3708496, 23.3732529
10: -16.7124729, 11.0089989, -16.7124729, 11.0089989, -25.4808578, 25.4849854
11: -5.9227352, 16.3983803, -5.9227352, 16.3983803, -17.6645279, 17.6642761
12: -22.6332550, 13.7408199, -22.6332550, 13.7408199, -29.3704224, 29.3673782
13: -33.4803886, 6.7535558, -33.4803886, 6.7535558, -30.3635941, 30.3589783
14: -36.9993134, 8.4883614, -36.9993134, 8.4883614, -43.5286560, 43.5306244
15: -17.1914825, 9.4443750, -17.1914825, 9.4443750, -24.1069641, 24.1078224
16: -19.6866646, 3.7832248, -19.6866646, 3.7832248, -20.9997711, 21.0060158
17: -26.4584827, 7.8025665, -26.4584827, 7.8025665, -34.2610474, 34.2610474
18: -7.6236534, 25.4448280, -7.6236534, 25.4448280, -31.1973267, 31.1972351
19: -1.0060940, 16.0898647, -1.0060940, 16.0898647, -15.6001873, 15.5995789
20: -7.0599399, 12.3405714, -7.0599399, 12.3405714, -18.5342979, 18.5352821
21: -5.5422239, 16.3245010, -5.5422239, 16.3245010, -21.0085754, 21.0086861
22: -2.6600771, 16.8842926, -2.6600771, 16.8842926, -16.6960564, 16.6924133
23: -4.0286174, 17.7371578, -4.0286174, 17.7371578, -18.8107605, 18.8092880
24: -2.7977962, 22.2555695, -2.7977962, 22.2555695, -21.7360458, 21.7345314
25: -5.3005190, 18.3410835, -5.3005190, 18.3410835, -21.0698318, 21.0667915
26: -7.8337917, 24.5785217, -7.8337917, 24.5785217, -30.5096512, 30.5109482
27: -5.9679432, 18.1709843, -5.9679432, 18.1709843, -20.6945953, 20.6946526
28: -2.9235144, 20.5516129, -2.9235144, 20.5516129, -21.6679611, 21.6676407
29: -2.4712782, 17.2291317, -2.4712782, 17.2291317, -15.6593018, 15.6579857
30: -9.8343687, 18.7539310, -9.8343687, 18.7539310, -26.2848816, 26.2844009
31: -5.5064917, 17.6745300, -5.5064917, 17.6745300, -21.6650620, 21.6629372
32: -28.7472782, -1.3468709, -28.7472782, -1.3468709, -21.0544472, 21.0515480
33: -50.9105721, -11.6670780, -50.9105721, -11.6670780, -27.8253098, 27.8187714
34: -45.2720490, -13.7597017, -45.2720490, -13.7597017, -24.4619446, 24.4583206
35: -32.3548393, -2.8940580, -32.3548393, -2.8940580, -23.2897530, 23.2850113
36: -29.4910297, 2.3401928, -29.4910297, 2.3401928, -25.4006119, 25.3951111
37: -46.4651604, -5.4649081, -46.4651604, -5.4649081, -36.1756134, 36.1728516
38: -40.1517830, -2.7374935, -40.1517830, -2.7374935, -32.8113556, 32.8065948
39: -50.4003029, -7.9412012, -50.4003029, -7.9412012, -29.5099716, 29.5032196
40: -48.0460854, -17.6561031, -48.0460854, -17.6561031, -25.0408936, 25.0418930
41: -28.8077698, 0.9408860, -28.8077698, 0.9408860, -25.7902451, 25.7901611
42: -32.4999657, -9.4782362, -32.4999657, -9.4782362, -18.9015045, 18.9026070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=240, inp2_unstable=240, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1336
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1364

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6925064, upper bound: 19.6796680
time: 37.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6830268, upper bound: 19.6891514
time: 29.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 68.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6885967, upper bound: 19.6727413
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6793058, upper bound: 19.6820337
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6813328, upper bound: 19.6879285
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6841304, upper bound: 19.6851309
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6871063, upper bound: 19.6882709
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6878912, upper bound: 19.6874855
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6885627, upper bound: 19.6865970
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6890780, upper bound: 19.6860840
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6908058, upper bound: 19.6761101
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6830276, upper bound: 19.6836558
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6891036, upper bound: 19.6854676
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6895593, upper bound: 19.6850109
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6742554, upper bound: 19.6570221
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6742554, upper bound: 19.6570221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6798278, upper bound: 19.6673537
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6934014, upper bound: 19.6537867
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6841752, upper bound: 19.6612229
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6775193, upper bound: 19.6678794
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6918331, upper bound: 19.6677907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6899287, upper bound: 19.6696846
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6938280, upper bound: 19.6687040
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6881754, upper bound: 19.6714535
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6893628, upper bound: 19.6740004
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6891843, upper bound: 19.6741797
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6727553, upper bound: 19.6828287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6639868, upper bound: 19.6915926
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6912250, upper bound: 19.6918768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6912329, upper bound: 19.6918684
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6925064, upper bound: 19.6796680
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 68.95
Output dim: 26, lower bound: -19.6830268, upper bound: 19.6891514
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6916028, upper bound: 19.6729654
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6735468, upper bound: 19.6910182
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6683722, upper bound: 19.6892301
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6947699, upper bound: 19.6633398
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6757200, upper bound: 19.6914094
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6789106, upper bound: 19.6888684
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6787881, upper bound: 19.6889895
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6777871, upper bound: 19.6877973
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 68.95
Output dim: 26, lower bound: -19.6777809, upper bound: 19.6878596

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 47.79 + 3589.91 = 3637.70 seconds
