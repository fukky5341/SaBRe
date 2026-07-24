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
execution time: IAR + RelationalAnalysis = 2.40 + 45.43 = 47.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 26, lower bound: -19.7062175, upper bound: 19.7062176

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1308

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1607

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6930864, upper bound: 19.6415027
time: 30.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7052723, upper bound: 19.7052719
time: 29.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 59.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 59.73
Output dim: 26, lower bound: -19.6930864, upper bound: 19.6415027
IS_A2, status: Status.UNKNOWN, split count: 1, time: 59.73
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

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1308

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6242992, upper bound: 19.6382823
time: 44.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6905570, upper bound: 19.6389698
time: 38.24 seconds

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

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1308

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6366857, upper bound: 19.7021267
time: 34.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7027539, upper bound: 19.7027536
time: 35.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 72.56 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 72.56
Output dim: 26, lower bound: -19.6242992, upper bound: 19.6382823
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 72.56
Output dim: 26, lower bound: -19.6905570, upper bound: 19.6389698
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 72.56
Output dim: 26, lower bound: -19.6366857, upper bound: 19.7021267
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 72.56
Output dim: 26, lower bound: -19.7027539, upper bound: 19.7027536

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -33.9180222, -0.6106336, -33.9224319, -0.6043844, -24.7138748, 24.6791534
1: -13.1139555, 7.4287882, -13.1211996, 7.4370375, -15.9990921, 15.9957809
2: -12.1595144, 8.6014595, -12.1648664, 8.6187258, -15.5099792, 15.4611130
3: -26.8507271, -2.2268167, -26.8750992, -2.2134466, -20.0420227, 20.0522461
4: -16.8615551, 11.0030289, -16.8649082, 11.0223799, -21.0124588, 20.9167480
5: -21.6327705, 3.2639780, -21.6339245, 3.2684603, -19.3787842, 19.3699074
6: -34.6313629, -7.5722547, -34.6764984, -7.5471969, -22.5667114, 22.5484848
7: -20.9211502, 6.2451644, -20.9270153, 6.2461052, -21.5033035, 21.4964523
8: -31.0351028, 4.9986925, -31.0370750, 5.0062551, -26.4361649, 26.4448318
9: -18.9395161, 8.0167599, -18.9652710, 8.0356636, -23.3251953, 23.3150406
10: -16.6475143, 10.9773464, -16.6802292, 11.0006809, -25.3882141, 25.4257278
11: -5.8967743, 16.3782177, -5.9114170, 16.3872833, -17.5502014, 17.6370430
12: -22.5476284, 13.6949425, -22.5915012, 13.7286148, -29.2215424, 29.2865448
13: -33.3705254, 6.7040086, -33.4229507, 6.7457938, -30.2522736, 30.2177963
14: -36.9562607, 8.4185867, -36.9855614, 8.4515142, -43.4417267, 43.4524689
15: -17.1779308, 9.4041214, -17.1853504, 9.4247322, -24.0825424, 24.0864944
16: -19.6463070, 3.7741442, -19.6668816, 3.7793124, -20.9754639, 20.9898872
17: -26.4135742, 7.7167521, -26.4486713, 7.7607012, -34.1742744, 34.1654243
18: -7.5678606, 25.3051643, -7.6146889, 25.3736973, -31.0678406, 31.0520935
19: -0.9659624, 16.0408859, -0.9953613, 16.0663700, -15.5359154, 15.5375862
20: -7.0380945, 12.3123474, -7.0530396, 12.3268251, -18.5245972, 18.5003471
21: -5.5083523, 16.2909355, -5.5299635, 16.3077145, -20.9561310, 20.9696198
22: -2.6287384, 16.8305264, -2.6534624, 16.8582592, -16.6347771, 16.6547546
23: -4.0007238, 17.6882343, -4.0214791, 17.7115936, -18.7410583, 18.7582932
24: -2.7515574, 22.1665039, -2.7895155, 22.2110825, -21.6481972, 21.6427269
25: -5.2707872, 18.2954483, -5.2922277, 18.3186150, -21.0073051, 21.0128059
26: -7.7806530, 24.4704552, -7.8244438, 24.5249672, -30.3789749, 30.3924103
27: -5.9175477, 18.0765018, -5.9597387, 18.1229572, -20.5963440, 20.5933533
28: -2.8858109, 20.4984436, -2.9162655, 20.5250206, -21.6101379, 21.6065750
29: -2.4388471, 17.1832848, -2.4649506, 17.2062740, -15.5835762, 15.6091690
30: -9.8087530, 18.6993637, -9.8248653, 18.7255287, -26.1995773, 26.2299728
31: -5.4645605, 17.6203651, -5.4946785, 17.6482697, -21.5998497, 21.6007271
32: -28.6231422, -1.3979864, -28.6874428, -1.3543491, -20.9279251, 20.9491119
33: -50.8211136, -11.7004490, -50.8661346, -11.6706619, -27.7410507, 27.6849060
34: -45.1782951, -13.7886963, -45.2275429, -13.7644348, -24.3894958, 24.4156647
35: -32.2865677, -2.9111443, -32.3215904, -2.8963497, -23.2343826, 23.2513847
36: -29.4366798, 2.3221846, -29.4652214, 2.3371739, -25.3504028, 25.3485260
37: -46.4162827, -5.4791589, -46.4393082, -5.4680195, -36.1282196, 36.1497498
38: -40.0944824, -2.7613778, -40.1234474, -2.7458696, -32.7681351, 32.7713928
39: -50.2945862, -7.9722905, -50.3472404, -7.9439907, -29.4009018, 29.3280106
40: -47.9760437, -17.6785831, -48.0098419, -17.6589241, -24.9790497, 24.9683075
41: -28.7192879, 0.9059501, -28.7661476, 0.9348068, -25.7057495, 25.7204971
42: -32.3922348, -9.5199738, -32.4491844, -9.4879608, -18.7629890, 18.8112335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=239, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1308

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6898782, upper bound: 19.5726415
time: 34.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6898782, upper bound: 19.6389698
time: 35.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -33.8913460, -0.6094816, -33.8276443, -0.6338933, -24.6620255, 24.6417847
1: -13.1220369, 7.4345889, -13.1150417, 7.4192500, -15.9860344, 15.9984741
2: -12.1277790, 8.6224442, -12.0552559, 8.5959597, -15.4541245, 15.4028091
3: -26.8678913, -2.2240763, -26.8144302, -2.2623205, -19.9769669, 19.9902000
4: -16.8041611, 11.0242233, -16.6828651, 10.9925070, -20.9227295, 20.8274117
5: -21.5892143, 3.2588143, -21.5001068, 3.2227378, -19.2824402, 19.2244873
6: -34.6996994, -7.5518589, -34.6957741, -7.5696621, -22.5137634, 22.5695572
7: -20.8835907, 6.2372093, -20.7923775, 6.2084332, -21.4260788, 21.3618240
8: -31.0202713, 5.0035238, -30.9816360, 4.9938283, -26.4264069, 26.3978577
9: -18.9681664, 8.0255728, -18.9243011, 8.0154400, -23.3049850, 23.3103180
10: -16.6953964, 10.9568310, -16.6581116, 10.8714447, -25.3093262, 25.3866425
11: -5.9097857, 16.3353462, -5.8662505, 16.2238369, -17.5049210, 17.5676079
12: -22.6177959, 13.6376781, -22.5813637, 13.4553280, -28.9936295, 29.1965485
13: -33.4358521, 6.7300744, -33.3591919, 6.7111344, -30.1796417, 30.2222137
14: -36.9557953, 8.3880825, -36.8779297, 8.2177572, -43.2285309, 43.3114624
15: -17.1755104, 9.4305439, -17.1499748, 9.4123888, -24.0630493, 24.0382004
16: -19.6667347, 3.7736087, -19.6412315, 3.7560472, -20.9723587, 20.9579506
17: -26.4366226, 7.6849985, -26.3897324, 7.4833145, -33.9199371, 34.0747299
18: -7.5944595, 25.4098587, -7.5790367, 25.3495846, -31.0852280, 31.0600281
19: -0.9810228, 16.0871487, -0.9373770, 16.0872326, -15.5694408, 15.4935760
20: -7.0379415, 12.3315563, -6.9993567, 12.3262444, -18.4863472, 18.4406128
21: -5.5214863, 16.3020172, -5.4751596, 16.2635403, -20.9403305, 20.9245453
22: -2.6420417, 16.8752403, -2.6140971, 16.8701305, -16.6572914, 16.6032391
23: -4.0122647, 17.7184525, -3.9913125, 17.6897888, -18.7587166, 18.7193146
24: -2.7744999, 22.2317886, -2.7517672, 22.1907387, -21.6459732, 21.6215820
25: -5.2811232, 18.3204498, -5.2424121, 18.2891998, -20.9932213, 20.9611626
26: -7.8059101, 24.5107479, -7.7417278, 24.3898373, -30.2906647, 30.3033600
27: -5.9383273, 18.1557064, -5.9099689, 18.1343727, -20.6182137, 20.5569839
28: -2.9027805, 20.5436993, -2.8628998, 20.5427666, -21.6317825, 21.5671921
29: -2.4559574, 17.1921444, -2.4276037, 17.1298904, -15.5438805, 15.5496063
30: -9.8205280, 18.7013474, -9.7850742, 18.6083622, -26.1527863, 26.2153854
31: -5.4745560, 17.6712570, -5.4274683, 17.6698723, -21.6297226, 21.5480309
32: -28.7055206, -1.3613701, -28.6352005, -1.3883791, -20.9028549, 20.9213905
33: -50.8474503, -11.6800394, -50.7428055, -11.7139683, -27.6521683, 27.6627579
34: -45.2541351, -13.7722435, -45.2360229, -13.7944565, -24.3806152, 24.4125671
35: -32.3245010, -2.9014707, -32.2818451, -2.9124894, -23.2495155, 23.1956253
36: -29.4506531, 2.3325515, -29.3880081, 2.3126025, -25.3033066, 25.2983551
37: -46.4297791, -5.4751096, -46.3816376, -5.4933171, -36.0859375, 36.0792542
38: -40.1216583, -2.7499499, -40.0922661, -2.7748313, -32.7083435, 32.7480545
39: -50.3133163, -7.9508920, -50.1663475, -7.9859586, -29.2933502, 29.2945480
40: -48.0076294, -17.6621590, -47.9472198, -17.6666336, -24.9521027, 24.9440536
41: -28.7844391, 0.9303951, -28.7507324, 0.9154425, -25.6997604, 25.7211380
42: -32.4888763, -9.5044117, -32.4924126, -9.5440207, -18.7689896, 18.8515129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=239, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1308

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6280470, upper bound: 19.6749484
time: 32.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.6280470, upper bound: 19.6936688
time: 35.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -33.9296112, -0.5989256, -33.9279671, -0.5991175, -24.7303162, 24.6960754
1: -13.1279869, 7.4422493, -13.1280384, 7.4420309, -16.0151978, 16.0166702
2: -12.1677094, 8.6318254, -12.1668472, 8.6319180, -15.5268059, 15.4834557
3: -26.8971252, -2.2068653, -26.8978367, -2.2073050, -20.0612411, 20.0941772
4: -16.8720131, 11.0353708, -16.8701878, 11.0370674, -21.0388069, 20.9531136
5: -21.6381645, 3.2736430, -21.6365986, 3.2732091, -19.3896255, 19.3840179
6: -34.7147217, -7.5421648, -34.7156677, -7.5432754, -22.6153564, 22.6188507
7: -20.9334831, 6.2498240, -20.9332047, 6.2493467, -21.5179825, 21.5078964
8: -31.0413780, 5.0138135, -31.0399075, 5.0136657, -26.4519272, 26.4643860
9: -18.9921722, 8.0389681, -18.9901047, 8.0387888, -23.3545074, 23.3647232
10: -16.7112522, 11.0085220, -16.7112160, 11.0062838, -25.4274826, 25.4891052
11: -5.9219060, 16.3972397, -5.9215946, 16.3964825, -17.5874519, 17.6687622
12: -22.6316700, 13.7401180, -22.6322250, 13.7366362, -29.2597504, 29.3667831
13: -33.4778900, 6.7527781, -33.4744225, 6.7519188, -30.2869644, 30.3146362
14: -36.9979477, 8.4866219, -36.9973488, 8.4834518, -43.5274963, 43.5310822
15: -17.1907654, 9.4424314, -17.1895447, 9.4430971, -24.1094818, 24.1195984
16: -19.6841354, 3.7829401, -19.6846275, 3.7827294, -21.0167084, 21.0109024
17: -26.4578781, 7.8006516, -26.4573421, 7.8002214, -34.2580986, 34.2579956
18: -7.6227036, 25.4428120, -7.6215525, 25.4392548, -31.1882324, 31.1337280
19: -1.0052118, 16.0889645, -1.0046234, 16.0894241, -15.5983219, 15.5639992
20: -7.0593233, 12.3394127, -7.0588732, 12.3396215, -18.5596199, 18.5199738
21: -5.5412884, 16.3232040, -5.5410748, 16.3234959, -21.0013428, 21.0055733
22: -2.6595240, 16.8826485, -2.6589680, 16.8834248, -16.6918449, 16.6884842
23: -4.0280218, 17.7358208, -4.0276880, 17.7346134, -18.7925110, 18.7864914
24: -2.7970343, 22.2537155, -2.7959719, 22.2525291, -21.7340126, 21.6934357
25: -5.2999167, 18.3398590, -5.2990232, 18.3396721, -21.0569458, 21.0380516
26: -7.8330994, 24.5759354, -7.8319340, 24.5755215, -30.4849167, 30.4605865
27: -5.9673033, 18.1689110, -5.9663882, 18.1673431, -20.6879807, 20.6371346
28: -2.9230394, 20.5503979, -2.9224796, 20.5504589, -21.6719437, 21.6373901
29: -2.4707279, 17.2279797, -2.4703212, 17.2279396, -15.6360092, 15.6291847
30: -9.8337355, 18.7519703, -9.8331490, 18.7514915, -26.2504501, 26.2889633
31: -5.5051861, 17.6737118, -5.5045166, 17.6741562, -21.6657944, 21.6303062
32: -28.7451267, -1.3473225, -28.7450180, -1.3481669, -20.9947968, 21.0539207
33: -50.9084778, -11.6675224, -50.9079208, -11.6676531, -27.7471161, 27.7573318
34: -45.2698593, -13.7600994, -45.2704849, -13.7604332, -24.4414902, 24.4571075
35: -32.3525620, -2.8943436, -32.3521118, -2.8944712, -23.2990799, 23.2673454
36: -29.4891720, 2.3399549, -29.4887371, 2.3398066, -25.3673782, 25.3904572
37: -46.4629822, -5.4651861, -46.4622231, -5.4652724, -36.1625443, 36.1777115
38: -40.1492767, -2.7379198, -40.1494827, -2.7396841, -32.7862091, 32.8230057
39: -50.3976822, -7.9414291, -50.3967667, -7.9416151, -29.4217834, 29.4130554
40: -48.0440063, -17.6563473, -48.0426788, -17.6564903, -25.0061722, 25.0108185
41: -28.8060799, 0.9405026, -28.8058872, 0.9401293, -25.7602768, 25.7925034
42: -32.4968681, -9.4789047, -32.4987488, -9.4824505, -18.8304977, 18.9016953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=239, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1308

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7021271, upper bound: 19.6366853
time: 35.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.7021271, upper bound: 19.7027536
time: 33.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 70.85 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 70.85
Output dim: 26, lower bound: -19.6898782, upper bound: 19.5726415
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 70.85
Output dim: 26, lower bound: -19.6898782, upper bound: 19.6389698
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 70.85
Output dim: 26, lower bound: -19.6280470, upper bound: 19.6749484
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 70.85
Output dim: 26, lower bound: -19.6280470, upper bound: 19.6936688
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 70.85
Output dim: 26, lower bound: -19.7021271, upper bound: 19.6366853
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 70.85
Output dim: 26, lower bound: -19.7021271, upper bound: 19.7027536

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -33.8152313, -0.6459050, -33.9224319, -0.6043844, -24.6364136, 24.6767197
1: -13.1004210, 7.4054489, -13.1211996, 7.4370375, -15.9858627, 15.9736404
2: -12.0467815, 8.5648117, -12.1648664, 8.6187258, -15.4027100, 15.4627075
3: -26.7659302, -2.2827539, -26.8750992, -2.2134466, -19.9517441, 19.9998932
4: -16.6719608, 10.9581776, -16.8649082, 11.0223799, -20.8147583, 20.9536705
5: -21.4943027, 3.2128024, -21.6339245, 3.2684603, -19.2276077, 19.3211975
6: -34.6105347, -7.5999813, -34.6764984, -7.5471969, -22.4889984, 22.5102730
7: -20.7786961, 6.2035155, -20.9270153, 6.2461052, -21.3551178, 21.4688568
8: -30.9753036, 4.9777822, -31.0370750, 5.0062551, -26.3957214, 26.4167328
9: -18.8708992, 7.9928784, -18.9652710, 8.0356636, -23.2624969, 23.3038559
10: -16.5936890, 10.8399525, -16.6802292, 11.0006809, -25.3621063, 25.2928047
11: -5.8405905, 16.2040253, -5.9114170, 16.3872833, -17.5901260, 17.4822617
12: -22.4962883, 13.4097042, -22.5915012, 13.7286148, -29.2142944, 28.9801025
13: -33.2501755, 6.6618214, -33.4229507, 6.7457938, -30.1333466, 30.2011337
14: -36.8350105, 8.1484528, -36.9855614, 8.4515142, -43.3301849, 43.1801453
15: -17.1366425, 9.3728943, -17.1853504, 9.4247322, -24.0335464, 24.0361824
16: -19.6016464, 3.7470555, -19.6668816, 3.7793124, -20.9286880, 20.9667206
17: -26.3447876, 7.3978386, -26.4486713, 7.7607012, -34.1054878, 33.8465118
18: -7.5234022, 25.2103977, -7.6146889, 25.3736973, -31.0271683, 30.9663162
19: -0.8974686, 16.0386505, -0.9953613, 16.0663700, -15.4629478, 15.5369759
20: -6.9776506, 12.2983685, -7.0530396, 12.3268251, -18.4251022, 18.4779129
21: -5.4416513, 16.2304211, -5.5299635, 16.3077145, -20.9027138, 20.9236374
22: -2.5829206, 16.8168278, -2.6534624, 16.8582592, -16.5834389, 16.6110306
23: -3.9635849, 17.6412601, -4.0214791, 17.7115936, -18.7126770, 18.7156792
24: -2.7057438, 22.1022224, -2.7895155, 22.2110825, -21.6027069, 21.5725174
25: -5.2128592, 18.2439690, -5.2922277, 18.3186150, -20.9568481, 20.9587440
26: -7.6886683, 24.2825470, -7.8244438, 24.5249672, -30.3109512, 30.2000656
27: -5.8598266, 18.0405140, -5.9597387, 18.1229572, -20.5340500, 20.5476913
28: -2.8253150, 20.4899998, -2.9162655, 20.5250206, -21.5440292, 21.5908051
29: -2.3953228, 17.0844021, -2.4649506, 17.2062740, -15.5643730, 15.5069046
30: -9.7595682, 18.5545464, -9.8248653, 18.7255287, -26.2103958, 26.1031723
31: -5.3859568, 17.6159554, -5.4946785, 17.6482697, -21.5168877, 21.5976067
32: -28.5116997, -1.4394779, -28.6874428, -1.3543491, -20.8030853, 20.9045143
33: -50.6539917, -11.7472620, -50.8661346, -11.6706619, -27.5928574, 27.7083969
34: -45.1430016, -13.8232737, -45.2275429, -13.7644348, -24.3502274, 24.3768845
35: -32.2143555, -2.9294589, -32.3215904, -2.8963497, -23.1592255, 23.2404709
36: -29.3342972, 2.2946544, -29.4652214, 2.3371739, -25.2485809, 25.3389587
37: -46.3335114, -5.5074935, -46.4393082, -5.4680195, -36.0517426, 36.0980606
38: -40.0358009, -2.7986293, -40.1234474, -2.7458696, -32.6970291, 32.7283630
39: -50.0612030, -8.0170403, -50.3472404, -7.9439907, -29.1932220, 29.3753433
40: -47.8778687, -17.6890392, -48.0098419, -17.6589241, -24.8871841, 24.9814606
41: -28.6625996, 0.8804812, -28.7661476, 0.9348068, -25.6386337, 25.6896057
42: -32.3857536, -9.5855665, -32.4491844, -9.4879608, -18.7687225, 18.7280083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1308

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1401

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5966487, upper bound: 19.5635796
time: 31.54 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6153591, upper bound: 19.5635796
time: 32.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -33.9154778, -0.6111436, -33.9224319, -0.6043844, -24.6789398, 24.6785812
1: -13.1134224, 7.4282136, -13.1211996, 7.4370375, -15.9965439, 15.9952507
2: -12.1584349, 8.6007681, -12.1648664, 8.6187258, -15.4730034, 15.4603767
3: -26.8492622, -2.2278199, -26.8750992, -2.2134466, -20.0409241, 20.0523758
4: -16.8592854, 11.0025826, -16.8649082, 11.0223799, -20.9301300, 20.9162827
5: -21.6307487, 3.2632780, -21.6339245, 3.2684603, -19.3715897, 19.3691406
6: -34.6304092, -7.5735798, -34.6764984, -7.5471969, -22.5646439, 22.5829201
7: -20.9194565, 6.2444291, -20.9270153, 6.2461052, -21.4893646, 21.4956970
8: -31.0335369, 4.9976077, -31.0370750, 5.0062551, -26.4495087, 26.4437256
9: -18.9367714, 8.0162630, -18.9652710, 8.0356636, -23.3052673, 23.3145447
10: -16.6466408, 10.9747448, -16.6802292, 11.0006809, -25.3873901, 25.3958588
11: -5.8958807, 16.3766956, -5.9114170, 16.3872833, -17.5493851, 17.5534172
12: -22.5470791, 13.6910038, -22.5915012, 13.7286148, -29.2208557, 29.2294922
13: -33.3653412, 6.7026539, -33.4229507, 6.7457938, -30.2007294, 30.2164459
14: -36.9547958, 8.4142027, -36.9855614, 8.4515142, -43.4402466, 43.4390106
15: -17.1762047, 9.4034081, -17.1853504, 9.4247322, -24.0983047, 24.0855255
16: -19.6450500, 3.7737246, -19.6668816, 3.7793124, -20.9711380, 20.9894447
17: -26.4126644, 7.7149677, -26.4486713, 7.7607012, -34.1733665, 34.1636391
18: -7.5660715, 25.3002052, -7.6146889, 25.3736973, -31.0660553, 31.0408478
19: -0.9647570, 16.0408192, -0.9953613, 16.0663700, -15.5320282, 15.5375290
20: -7.0372229, 12.3117418, -7.0530396, 12.3268251, -18.5229874, 18.5219345
21: -5.5075121, 16.2903175, -5.5299635, 16.3077145, -20.9551086, 20.9581528
22: -2.6278276, 16.8301697, -2.6534624, 16.8582592, -16.6556587, 16.6539612
23: -3.9999557, 17.6861305, -4.0214791, 17.7115936, -18.7403336, 18.7343140
24: -2.7499323, 22.1640091, -2.7895155, 22.2110825, -21.6466255, 21.6383514
25: -5.2694602, 18.2944298, -5.2922277, 18.3186150, -21.0060387, 21.0040970
26: -7.7790451, 24.4682426, -7.8244438, 24.5249672, -30.3773727, 30.3672485
27: -5.9161801, 18.0735016, -5.9597387, 18.1229572, -20.5950012, 20.5879173
28: -2.8849277, 20.4977093, -2.9162655, 20.5250206, -21.6091080, 21.6076660
29: -2.4380784, 17.1824493, -2.4649506, 17.2062740, -15.5828400, 15.5853119
30: -9.8076859, 18.6975460, -9.8248653, 18.7255287, -26.1986008, 26.1861954
31: -5.4630117, 17.6202240, -5.4946785, 17.6482697, -21.5966339, 21.6006088
32: -28.6214943, -1.3991179, -28.6874428, -1.3543491, -20.9269180, 20.9524612
33: -50.8190308, -11.7008991, -50.8661346, -11.6706619, -27.6738586, 27.6844177
34: -45.1774063, -13.7892666, -45.2275429, -13.7644348, -24.3885727, 24.4209061
35: -32.2845535, -2.9114597, -32.3215904, -2.8963497, -23.2228317, 23.2510300
36: -29.4348869, 2.3218632, -29.4652214, 2.3371739, -25.3331223, 25.3481369
37: -46.4140701, -5.4794602, -46.4393082, -5.4680195, -36.1312561, 36.1461487
38: -40.0929947, -2.7634068, -40.1234474, -2.7458696, -32.7665787, 32.7774506
39: -50.2918777, -7.9726171, -50.3472404, -7.9439907, -29.2994766, 29.3274994
40: -47.9732780, -17.6788940, -48.0098419, -17.6589241, -24.9467773, 24.9674988
41: -28.7179413, 0.9053316, -28.7661476, 0.9348068, -25.7041550, 25.7239761
42: -32.3919182, -9.5240078, -32.4491844, -9.4879608, -18.7627373, 18.7836189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1308

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1401

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5966487, upper bound: 19.6302353
time: 40.96 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.6153591, upper bound: 19.6302353
time: 30.38 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -33.8876190, -0.5754838, -33.8227997, -0.6353912, -24.6556244, 24.6680260
1: -13.1217556, 7.4548635, -13.1137066, 7.4183836, -15.9866333, 16.0165749
2: -12.1217299, 8.6529179, -12.0509090, 8.5944710, -15.4463081, 15.4294052
3: -26.8541260, -2.1793571, -26.8059387, -2.2652292, -19.9604874, 20.0272675
4: -16.7923737, 11.0588942, -16.6734867, 10.9907198, -20.9060287, 20.8539009
5: -21.5768394, 3.2896967, -21.4923782, 3.2200437, -19.2644348, 19.2484589
6: -34.7228584, -7.5584249, -34.6944351, -7.5765591, -22.5286331, 22.5635681
7: -20.8757858, 6.2701750, -20.7867489, 6.2063761, -21.4120102, 21.3861542
8: -31.0137634, 5.0277700, -30.9766083, 4.9924912, -26.4187698, 26.4104309
9: -18.9609642, 8.0599728, -18.9159794, 8.0132008, -23.2960587, 23.3417206
10: -16.7003593, 10.9874763, -16.6508598, 10.8680468, -25.3134842, 25.4113350
11: -5.9439101, 16.3276691, -5.8647223, 16.2182655, -17.5356903, 17.5591393
12: -22.6614094, 13.6153078, -22.5780754, 13.4395199, -29.0213013, 29.1676407
13: -33.4342918, 6.7561102, -33.3537140, 6.7082148, -30.1829529, 30.2371521
14: -36.9912796, 8.3852720, -36.8706131, 8.2120342, -43.2606812, 43.2999725
15: -17.1718159, 9.4658337, -17.1433086, 9.4107552, -24.0585098, 24.0641327
16: -19.6859398, 3.7989645, -19.6392803, 3.7546949, -20.9939423, 20.9785881
17: -26.4872818, 7.6578856, -26.3870640, 7.4637938, -33.9510765, 34.0449486
18: -7.6302681, 25.4005795, -7.5748682, 25.3407555, -31.1112366, 31.0479355
19: -1.0018787, 16.0878277, -0.9346085, 16.0868759, -15.5854073, 15.4972229
20: -7.0519652, 12.3371449, -6.9966974, 12.3250656, -18.4799156, 18.4533768
21: -5.5398397, 16.3027458, -5.4724483, 16.2624741, -20.9447784, 20.9314766
22: -2.6574955, 16.8807316, -2.6117969, 16.8690815, -16.6724167, 16.6060085
23: -4.0351877, 17.7176304, -3.9894757, 17.6888618, -18.7820930, 18.7166328
24: -2.7968721, 22.2307243, -2.7490788, 22.1879501, -21.6663132, 21.6173668
25: -5.3069072, 18.3204803, -5.2396808, 18.2874889, -21.0165062, 20.9585686
26: -7.8620443, 24.5033073, -7.7375441, 24.3824711, -30.3409271, 30.2906647
27: -5.9633193, 18.1506329, -5.9062395, 18.1296120, -20.6408424, 20.5490570
28: -2.9296255, 20.5422173, -2.8604002, 20.5411263, -21.6540451, 21.5646057
29: -2.4966888, 17.1900749, -2.4258957, 17.1276474, -15.5836334, 15.5438690
30: -9.8446627, 18.6957321, -9.7838535, 18.6037750, -26.1703415, 26.2113266
31: -5.4905548, 17.6778049, -5.4240170, 17.6693344, -21.6345444, 21.5579796
32: -28.7122993, -1.3400559, -28.6321697, -1.3909860, -20.9042778, 20.9432220
33: -50.8444786, -11.6372690, -50.7374954, -11.7163849, -27.6469002, 27.6990814
34: -45.2583961, -13.7483282, -45.2316666, -13.7960663, -24.3910675, 24.4148102
35: -32.3246307, -2.8925645, -32.2760963, -2.9139647, -23.2545319, 23.1822128
36: -29.4707718, 2.3303781, -29.3848610, 2.3095675, -25.3242340, 25.2860031
37: -46.4508667, -5.4803286, -46.3771248, -5.4991999, -36.1226120, 36.0461502
38: -40.1361771, -2.7557478, -40.0876694, -2.7810769, -32.7241669, 32.7332687
39: -50.3148270, -7.9127755, -50.1597672, -7.9872026, -29.2887802, 29.3204346
40: -48.0101318, -17.6610165, -47.9449158, -17.6679230, -24.9576263, 24.9282990
41: -28.7990208, 0.9327564, -28.7486992, 0.9134088, -25.7136383, 25.7144852
42: -32.4994011, -9.5024796, -32.4913788, -9.5479813, -18.7835541, 18.8485680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1607

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5635796, upper bound: 19.6812079
time: 34.66 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5635796, upper bound: 19.6936688
time: 33.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -33.8268280, -0.6341968, -33.9279671, -0.5991175, -24.6528244, 24.6935806
1: -13.1144257, 7.4189177, -13.1280384, 7.4420309, -16.0019531, 15.9945030
2: -12.0550442, 8.5951824, -12.1668472, 8.6319180, -15.4194679, 15.4849930
3: -26.8122711, -2.2628927, -26.8978367, -2.2073050, -19.9708939, 20.0417824
4: -16.6823978, 10.9903650, -16.8701878, 11.0370674, -20.8412971, 20.9899025
5: -21.4996529, 3.2224736, -21.6365986, 3.2732091, -19.2384109, 19.3353119
6: -34.6938782, -7.5698776, -34.7156677, -7.5432754, -22.5376892, 22.5806770
7: -20.7909622, 6.2082024, -20.9332047, 6.2493467, -21.3697510, 21.4804001
8: -30.9815102, 4.9929166, -31.0399075, 5.0136657, -26.4114227, 26.4362869
9: -18.9236641, 8.0151148, -18.9901047, 8.0387888, -23.2919083, 23.3535004
10: -16.6572876, 10.8711166, -16.7112160, 11.0062838, -25.4012451, 25.3560829
11: -5.8656921, 16.2230549, -5.9215946, 16.3964825, -17.6272697, 17.5139694
12: -22.5802441, 13.4548302, -22.6322250, 13.7366362, -29.2524796, 29.0603104
13: -33.3574867, 6.7106071, -33.4744225, 6.7519188, -30.1680222, 30.2980118
14: -36.8769798, 8.2164516, -36.9973488, 8.4834518, -43.4163208, 43.2588501
15: -17.1494751, 9.4110584, -17.1895447, 9.4430971, -24.0605164, 24.0693054
16: -19.6394768, 3.7558613, -19.6846275, 3.7827294, -20.9699783, 20.9877319
17: -26.3893032, 7.4820061, -26.4573421, 7.8002214, -34.1895256, 33.9393463
18: -7.5783777, 25.3481789, -7.6215525, 25.4392548, -31.1477356, 31.0479813
19: -0.9367628, 16.0867405, -1.0046234, 16.0894241, -15.5253353, 15.5633774
20: -6.9989223, 12.3254433, -7.0588732, 12.3396215, -18.4600830, 18.4974861
21: -5.4745083, 16.2626343, -5.5410748, 16.3234959, -20.9479370, 20.9594841
22: -2.6136880, 16.8689690, -2.6589680, 16.8834248, -16.6404915, 16.6448250
23: -3.9908991, 17.6888638, -4.0276880, 17.7346134, -18.7640877, 18.7438431
24: -2.7512398, 22.1894608, -2.7959719, 22.2525291, -21.6885223, 21.6232033
25: -5.2420049, 18.2883606, -5.2990232, 18.3396721, -21.0065231, 20.9839935
26: -7.7413025, 24.3880482, -7.8319340, 24.5755215, -30.4170456, 30.2683258
27: -5.9095216, 18.1329155, -5.9663882, 18.1673431, -20.6256866, 20.5914917
28: -2.8625603, 20.5419846, -2.9224796, 20.5504589, -21.6058807, 21.6216202
29: -2.4272127, 17.1290970, -2.4703212, 17.2279396, -15.6168480, 15.5268784
30: -9.7846289, 18.6070805, -9.8331490, 18.7514915, -26.2612076, 26.1619949
31: -5.4265728, 17.6692982, -5.5045166, 17.6741562, -21.5827103, 21.6271706
32: -28.6337185, -1.3887010, -28.7450180, -1.3481669, -20.8699951, 21.0094261
33: -50.7413445, -11.7142429, -50.9079208, -11.6676531, -27.5989227, 27.7808609
34: -45.2345085, -13.7947598, -45.2704849, -13.7604332, -24.4021988, 24.4183617
35: -32.2803192, -2.9127054, -32.3521118, -2.8944712, -23.2238846, 23.2564735
36: -29.3867626, 2.3124442, -29.4887371, 2.3398066, -25.2655563, 25.3808594
37: -46.3801651, -5.4934969, -46.4622231, -5.4652724, -36.0859909, 36.1257858
38: -40.0906067, -2.7751503, -40.1494827, -2.7396841, -32.7150269, 32.7800293
39: -50.1645126, -7.9861112, -50.3967667, -7.9416151, -29.2145233, 29.4602966
40: -47.9457932, -17.6667423, -48.0426788, -17.6564903, -24.9143906, 25.0239868
41: -28.7495308, 0.9151769, -28.8058872, 0.9401293, -25.6932602, 25.7618179
42: -32.4902802, -9.5444717, -32.4987488, -9.4824505, -18.8361664, 18.8185043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1308

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1607

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6242989
time: 43.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6242993
time: 37.39 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -33.9270821, -0.5994694, -33.9279671, -0.5991175, -24.6953583, 24.6954956
1: -13.1274347, 7.4416504, -13.1280384, 7.4420309, -16.0126457, 16.0161476
2: -12.1666441, 8.6311321, -12.1668472, 8.6319180, -15.4898300, 15.4827118
3: -26.8956680, -2.2078309, -26.8978367, -2.2073050, -20.0601273, 20.0942993
4: -16.8697338, 11.0349178, -16.8701878, 11.0370674, -20.9564896, 20.9526558
5: -21.6361732, 3.2728982, -21.6365986, 3.2732091, -19.3824234, 19.3832130
6: -34.7137680, -7.5434995, -34.7156677, -7.5432754, -22.6132812, 22.6533051
7: -20.9317722, 6.2490997, -20.9332047, 6.2493467, -21.5040359, 21.5071793
8: -31.0398102, 5.0127401, -31.0399075, 5.0136657, -26.4652481, 26.4632950
9: -18.9894638, 8.0385189, -18.9901047, 8.0387888, -23.3346939, 23.3641891
10: -16.7103825, 11.0059261, -16.7112160, 11.0062838, -25.4266357, 25.4591827
11: -5.9210210, 16.3957005, -5.9215946, 16.3964825, -17.5866432, 17.5851212
12: -22.6311111, 13.7361946, -22.6322250, 13.7366362, -29.2590332, 29.3097229
13: -33.4726868, 6.7513885, -33.4744225, 6.7519188, -30.2353973, 30.3133545
14: -36.9964066, 8.4822388, -36.9973488, 8.4834518, -43.5260315, 43.5176239
15: -17.1890678, 9.4417753, -17.1895447, 9.4430971, -24.1252136, 24.1186523
16: -19.6828728, 3.7825260, -19.6846275, 3.7827294, -21.0123749, 21.0104561
17: -26.4569016, 7.7988844, -26.4573421, 7.8002214, -34.2571220, 34.2562256
18: -7.6209164, 25.4378738, -7.6215525, 25.4392548, -31.1864777, 31.1224747
19: -1.0039997, 16.0889091, -1.0046234, 16.0894241, -15.5944366, 15.5639420
20: -7.0584550, 12.3388357, -7.0588732, 12.3396215, -18.5580025, 18.5415497
21: -5.5404329, 16.3225937, -5.5410748, 16.3234959, -21.0003204, 20.9940567
22: -2.6586018, 16.8823071, -2.6589680, 16.8834248, -16.7127151, 16.6877079
23: -4.0272841, 17.7337017, -4.0276880, 17.7346134, -18.7917709, 18.7625084
24: -2.7954082, 22.2512283, -2.7959719, 22.2525291, -21.7324562, 21.6890526
25: -5.2985692, 18.3388233, -5.2990232, 18.3396721, -21.0556717, 21.0293503
26: -7.8315048, 24.5737228, -7.8319340, 24.5755215, -30.4833145, 30.4354248
27: -5.9659629, 18.1658878, -5.9663882, 18.1673431, -20.6866455, 20.6317101
28: -2.9221573, 20.5496712, -2.9224796, 20.5504589, -21.6709442, 21.6384850
29: -2.4699483, 17.2271404, -2.4703212, 17.2279396, -15.6352692, 15.6053238
30: -9.8326855, 18.7501526, -9.8331490, 18.7514915, -26.2494659, 26.2451553
31: -5.5036354, 17.6735821, -5.5045166, 17.6741562, -21.6625671, 21.6301842
32: -28.7435570, -1.3484392, -28.7450180, -1.3481669, -20.9938126, 21.0572472
33: -50.9064255, -11.6679163, -50.9079208, -11.6676531, -27.6798706, 27.7568207
34: -45.2689743, -13.7606640, -45.2704849, -13.7604332, -24.4405899, 24.4623413
35: -32.3505402, -2.8946435, -32.3521118, -2.8944712, -23.2875099, 23.2669716
36: -29.4874649, 2.3396106, -29.4887371, 2.3398066, -25.3500595, 25.3900757
37: -46.4607468, -5.4654737, -46.4622231, -5.4652724, -36.1655731, 36.1740952
38: -40.1477661, -2.7399821, -40.1494827, -2.7396841, -32.7846527, 32.8290329
39: -50.3949356, -7.9417405, -50.3967667, -7.9416151, -29.3206100, 29.4125671
40: -48.0412788, -17.6566582, -48.0426788, -17.6564903, -24.9739227, 25.0100098
41: -28.8047543, 0.9398665, -28.8058872, 0.9401293, -25.7586517, 25.7959824
42: -32.4966202, -9.4829187, -32.4987488, -9.4824505, -18.8302536, 18.8740921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=239, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1308

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1607

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6905567
time: 37.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6905572
time: 31.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 71.73 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5966487, upper bound: 19.5635796
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.6153591, upper bound: 19.5635796
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5966487, upper bound: 19.6302353
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.6153591, upper bound: 19.6302353
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5635796, upper bound: 19.6812079
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5635796, upper bound: 19.6936688
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6242989
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6242993
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6905567
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 71.73
Output dim: 26, lower bound: -19.5726415, upper bound: 19.6905572

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -33.8876190, -0.5754838, -33.8219872, -0.6357760, -24.6549759, 24.6671715
1: -13.1217556, 7.4548635, -13.1130848, 7.4180527, -15.9863129, 16.0127716
2: -12.1217299, 8.6529179, -12.0506811, 8.5936823, -15.4390106, 15.4291725
3: -26.8541260, -2.1793571, -26.8037701, -2.2657933, -19.9599228, 19.9925461
4: -16.7923737, 11.0588942, -16.6730232, 10.9885921, -20.9017601, 20.8534584
5: -21.5768394, 3.2896967, -21.4919434, 3.2197752, -19.2646179, 19.2478638
6: -34.7228584, -7.5584249, -34.6925163, -7.5767546, -22.5284195, 22.5233498
7: -20.8757858, 6.2701750, -20.7853394, 6.2060943, -21.4117661, 21.3827705
8: -31.0137634, 5.0277700, -30.9764805, 4.9916072, -26.4166336, 26.4102859
9: -18.9609642, 8.0599728, -18.9153137, 8.0128908, -23.2958374, 23.3119812
10: -16.7003593, 10.9874763, -16.6500282, 10.8677092, -25.3131714, 25.3784943
11: -5.9439101, 16.3276691, -5.8641415, 16.2175045, -17.5349426, 17.5598907
12: -22.6614094, 13.6153078, -22.5769691, 13.4390011, -29.0208817, 29.1165314
13: -33.4342918, 6.7561102, -33.3520088, 6.7076349, -30.1824493, 30.1586914
14: -36.9912796, 8.3852720, -36.8696098, 8.2107868, -43.2593536, 43.3069916
15: -17.1718159, 9.4658337, -17.1428223, 9.4094219, -24.0514679, 24.0636520
16: -19.6859398, 3.7989645, -19.6375160, 3.7544701, -20.9935989, 20.9801331
17: -26.4872818, 7.6578856, -26.3866100, 7.4624233, -33.9497070, 34.0444946
18: -7.6302681, 25.4005795, -7.5741997, 25.3393497, -31.0466156, 31.0473251
19: -1.0018787, 16.0878277, -0.9339733, 16.0863724, -15.5543365, 15.4966412
20: -7.0519652, 12.3371449, -6.9962687, 12.3242826, -18.4630508, 18.4529419
21: -5.5398397, 16.3027458, -5.4717951, 16.2616158, -20.9379272, 20.9309006
22: -2.6574955, 16.8807316, -2.6114020, 16.8679619, -16.6470680, 16.6056614
23: -4.0351877, 17.7176304, -3.9890580, 17.6879349, -18.7524109, 18.7162018
24: -2.7968721, 22.2307243, -2.7485867, 22.1866493, -21.6223793, 21.6167946
25: -5.3069072, 18.3204803, -5.2392788, 18.2866306, -20.9896927, 20.9581108
26: -7.8620443, 24.5033073, -7.7370720, 24.3806858, -30.2925720, 30.2901764
27: -5.9633193, 18.1506329, -5.9057889, 18.1281776, -20.5855141, 20.5486450
28: -2.9296255, 20.5422173, -2.8600693, 20.5403118, -21.6212158, 21.5642586
29: -2.4966888, 17.1900749, -2.4255152, 17.1268349, -15.5533142, 15.5434647
30: -9.8446627, 18.6957321, -9.7834530, 18.6024666, -26.1656723, 26.2109375
31: -5.4905548, 17.6778049, -5.4231253, 17.6687546, -21.6013107, 21.5571136
32: -28.7122993, -1.3400559, -28.6306763, -1.3912697, -20.9040108, 20.8795090
33: -50.8444786, -11.6372690, -50.7360764, -11.7166023, -27.6464615, 27.6217575
34: -45.2583961, -13.7483282, -45.2301216, -13.7963581, -24.3897247, 24.3918877
35: -32.3246307, -2.8925645, -32.2744751, -2.9141381, -23.2536240, 23.2020302
36: -29.4707718, 2.3303781, -29.3835831, 2.3093662, -25.3240509, 25.2458801
37: -46.4508667, -5.4803286, -46.3756256, -5.4993558, -36.1208191, 36.0358353
38: -40.1361771, -2.7557478, -40.0859299, -2.7813349, -32.7238235, 32.6884918
39: -50.3148270, -7.9127755, -50.1579018, -7.9873734, -29.2885590, 29.2282181
40: -48.0101318, -17.6610165, -47.9435043, -17.6680622, -24.9574661, 24.8920593
41: -28.7990208, 0.9327564, -28.7474957, 0.9131370, -25.7133484, 25.6768951
42: -32.4994011, -9.5024796, -32.4892502, -9.5484295, -18.7831383, 18.8043098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=238, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1532

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.4991668, upper bound: 19.6904203
time: 49.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5632743, upper bound: 19.6806828
time: 40.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -33.9270821, -0.5994694, -33.9154778, -0.6111436, -24.6829910, 24.6839790
1: -13.1274347, 7.4416504, -13.1134224, 7.4282136, -16.0020218, 16.0007668
2: -12.1666441, 8.6311321, -12.1584349, 8.6007681, -15.4624863, 15.4831276
3: -26.8956680, -2.2078309, -26.8492622, -2.2278199, -20.0729218, 20.0465240
4: -16.8697338, 11.0349178, -16.8592854, 11.0025826, -20.9220123, 20.9423103
5: -21.6361732, 3.2728982, -21.6307487, 3.2632780, -19.3719406, 19.3759270
6: -34.7137680, -7.5434995, -34.6304092, -7.5735798, -22.6199036, 22.5685196
7: -20.9317722, 6.2490997, -20.9194565, 6.2444291, -21.5014114, 21.4923248
8: -31.0398102, 5.0127401, -31.0335369, 4.9976077, -26.4467621, 26.4573288
9: -18.9894638, 8.0385189, -18.9367714, 8.0162630, -23.3407288, 23.3080902
10: -16.7103825, 11.0059261, -16.6466408, 10.9747448, -25.4267197, 25.3925323
11: -5.9210210, 16.3957005, -5.8958807, 16.3766956, -17.5613708, 17.5603867
12: -22.6311111, 13.7361946, -22.5470791, 13.6910038, -29.2681808, 29.2272339
13: -33.4726868, 6.7513885, -33.3653412, 6.7026539, -30.2657623, 30.2060699
14: -36.9964066, 8.4822388, -36.9547958, 8.4142027, -43.4470978, 43.4741211
15: -17.1890678, 9.4417753, -17.1762047, 9.4034081, -24.0903091, 24.1124306
16: -19.6828728, 3.7825260, -19.6450500, 3.7737246, -21.0016327, 20.9764023
17: -26.4569016, 7.7988844, -26.4126644, 7.7149677, -34.1718674, 34.2115479
18: -7.6209164, 25.4378738, -7.5660715, 25.3002052, -31.0471497, 31.1298065
19: -1.0039997, 16.0889091, -0.9647570, 16.0408192, -15.5455093, 15.5553017
20: -7.0584550, 12.3388357, -7.0372229, 12.3117418, -18.5270348, 18.5362930
21: -5.5404329, 16.3225937, -5.5075121, 16.2903175, -20.9668884, 20.9703102
22: -2.6586018, 16.8823071, -2.6278276, 16.8301697, -16.6589813, 16.6804428
23: -4.0272841, 17.7337017, -3.9999557, 17.6861305, -18.7402649, 18.7639275
24: -2.7954082, 22.2512283, -2.7499323, 22.1640091, -21.6437454, 21.6869965
25: -5.2985692, 18.3388233, -5.2694602, 18.2944298, -21.0106468, 21.0265732
26: -7.8315048, 24.5737228, -7.7790451, 24.4682426, -30.3750000, 30.4274292
27: -5.9659629, 18.1658878, -5.9161801, 18.0735016, -20.5934067, 20.6371117
28: -2.9221573, 20.5496712, -2.8849277, 20.4977093, -21.6134796, 21.6354179
29: -2.4699483, 17.2271404, -2.4380784, 17.1824493, -15.5900002, 15.6037598
30: -9.8326855, 18.7501526, -9.8076859, 18.6975460, -26.1924820, 26.2245789
31: -5.5036354, 17.6735821, -5.4630117, 17.6202240, -21.6092262, 21.6217918
32: -28.7435570, -1.3484392, -28.6214943, -1.3991179, -21.0090179, 20.9320984
33: -50.9064255, -11.6679163, -50.8190308, -11.7008991, -27.7254524, 27.6765594
34: -45.2689743, -13.7606640, -45.1774063, -13.7892666, -24.4543457, 24.3902740
35: -32.3505402, -2.8946435, -32.2845535, -2.9114597, -23.2671852, 23.2220459
36: -29.4874649, 2.3396106, -29.4348869, 2.3218632, -25.3720474, 25.3355942
37: -46.4607468, -5.4654737, -46.4140701, -5.4794602, -36.1635895, 36.1337051
38: -40.1477661, -2.7399821, -40.0929947, -2.7634068, -32.8033066, 32.7723618
39: -50.3949356, -7.9417405, -50.2918777, -7.9726171, -29.3786621, 29.3021317
40: -48.0412788, -17.6566582, -47.9732780, -17.6788940, -24.9954834, 24.9495087
41: -28.8047543, 0.9398665, -28.7179413, 0.9053316, -25.7623825, 25.7089462
42: -32.4966202, -9.4829187, -32.3919182, -9.5240078, -18.8317680, 18.7668076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=238, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1308

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5682124, upper bound: 19.6632283
time: 32.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5682125, upper bound: 19.6818741
time: 35.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -33.9270821, -0.5994694, -33.9270821, -0.5994694, -24.6946487, 24.6946449
1: -13.1274347, 7.4416504, -13.1274347, 7.4416504, -16.0123367, 16.0123367
2: -12.1666441, 8.6311321, -12.1666441, 8.6311321, -15.4824791, 15.4824829
3: -26.8956680, -2.2078309, -26.8956680, -2.2078309, -20.0595627, 20.0595665
4: -16.8697338, 11.0349178, -16.8697338, 11.0349178, -20.9522209, 20.9522171
5: -21.6361732, 3.2728982, -21.6361732, 3.2728982, -19.3826141, 19.3826141
6: -34.7137680, -7.5434995, -34.7137680, -7.5434995, -22.6130753, 22.6130753
7: -20.9317722, 6.2490997, -20.9317722, 6.2490997, -21.5037994, 21.5037956
8: -31.0398102, 5.0127401, -31.0398102, 5.0127401, -26.4631348, 26.4631348
9: -18.9894638, 8.0385189, -18.9894638, 8.0385189, -23.3344574, 23.3344574
10: -16.7103825, 11.0059261, -16.7103825, 11.0059261, -25.4262924, 25.4262886
11: -5.9210210, 16.3957005, -5.9210210, 16.3957005, -17.5858765, 17.5858765
12: -22.6311111, 13.7361946, -22.6311111, 13.7361946, -29.2585907, 29.2585831
13: -33.4726868, 6.7513885, -33.4726868, 6.7513885, -30.2349014, 30.2349014
14: -36.9964066, 8.4822388, -36.9964066, 8.4822388, -43.5246735, 43.5246735
15: -17.1890678, 9.4417753, -17.1890678, 9.4417753, -24.1181793, 24.1181793
16: -19.6828728, 3.7825260, -19.6828728, 3.7825260, -21.0120010, 21.0120010
17: -26.4569016, 7.7988844, -26.4569016, 7.7988844, -34.2557869, 34.2557869
18: -7.6209164, 25.4378738, -7.6209164, 25.4378738, -31.1218643, 31.1218567
19: -1.0039997, 16.0889091, -1.0039997, 16.0889091, -15.5633659, 15.5633621
20: -7.0584550, 12.3388357, -7.0584550, 12.3388357, -18.5411148, 18.5411148
21: -5.5404329, 16.3225937, -5.5404329, 16.3225937, -20.9934959, 20.9934998
22: -2.6586018, 16.8823071, -2.6586018, 16.8823071, -16.6873589, 16.6873569
23: -4.0272841, 17.7337017, -4.0272841, 17.7337017, -18.7620850, 18.7620811
24: -2.7954082, 22.2512283, -2.7954082, 22.2512283, -21.6885185, 21.6885185
25: -5.2985692, 18.3388233, -5.2985692, 18.3388233, -21.0289001, 21.0288925
26: -7.8315048, 24.5737228, -7.8315048, 24.5737228, -30.4349823, 30.4349823
27: -5.9659629, 18.1658878, -5.9659629, 18.1658878, -20.6313095, 20.6313095
28: -2.9221573, 20.5496712, -2.9221573, 20.5496712, -21.6381226, 21.6381226
29: -2.4699483, 17.2271404, -2.4699483, 17.2271404, -15.6049423, 15.6049423
30: -9.8326855, 18.7501526, -9.8326855, 18.7501526, -26.2447739, 26.2447739
31: -5.5036354, 17.6735821, -5.5036354, 17.6735821, -21.6293335, 21.6293411
32: -28.7435570, -1.3484392, -28.7435570, -1.3484392, -20.9935303, 20.9935303
33: -50.9064255, -11.6679163, -50.9064255, -11.6679163, -27.6793976, 27.6793976
34: -45.2689743, -13.7606640, -45.2689743, -13.7606640, -24.4392395, 24.4392395
35: -32.3505402, -2.8946435, -32.3505402, -2.8946435, -23.2866058, 23.2866058
36: -29.4874649, 2.3396106, -29.4874649, 2.3396106, -25.3498840, 25.3498840
37: -46.4607468, -5.4654737, -46.4607468, -5.4654737, -36.1638794, 36.1638870
38: -40.1477661, -2.7399821, -40.1477661, -2.7399821, -32.7843246, 32.7843170
39: -50.3949356, -7.9417405, -50.3949356, -7.9417405, -29.3204193, 29.3204193
40: -48.0412788, -17.6566582, -48.0412788, -17.6566582, -24.9737778, 24.9737778
41: -28.8047543, 0.9398665, -28.8047543, 0.9398665, -25.7583771, 25.7583771
42: -32.4966202, -9.4829187, -32.4966202, -9.4829187, -18.8298569, 18.8298569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=238, inp2_unstable=238, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=165, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1308

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5682125, upper bound: 19.6632288
time: 42.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5682125, upper bound: 19.6818746
time: 37.27 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 81.83 seconds
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 81.83
Output dim: 26, lower bound: -19.4991668, upper bound: 19.6904203
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 81.83
Output dim: 26, lower bound: -19.5632743, upper bound: 19.6806828
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 81.83
Output dim: 26, lower bound: -19.5682124, upper bound: 19.6632283
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 81.83
Output dim: 26, lower bound: -19.5682125, upper bound: 19.6818741
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 81.83
Output dim: 26, lower bound: -19.5682125, upper bound: 19.6632288
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 81.83
Output dim: 26, lower bound: -19.5682125, upper bound: 19.6818746

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -33.8675690, -0.6214929, -33.8179550, -0.6579590, -24.6183853, 24.6159096
1: -13.1154442, 7.4122591, -13.1122704, 7.3976679, -15.9669304, 15.9673653
2: -12.1049843, 8.6005888, -12.0487251, 8.5686474, -15.3940926, 15.3736115
3: -26.8349113, -2.2435961, -26.8020191, -2.2959728, -19.9104767, 19.9271545
4: -16.7720032, 11.0115404, -16.6710606, 10.9667587, -20.8576813, 20.8049393
5: -21.5542126, 3.2302456, -21.4904366, 3.1919303, -19.2106094, 19.1838799
6: -34.7060165, -7.5795746, -34.6843605, -7.5859561, -22.4995956, 22.4911880
7: -20.8462067, 6.1961436, -20.7832127, 6.1709881, -21.3464584, 21.3067169
8: -31.0018730, 4.9910073, -30.9752636, 4.9739199, -26.3980179, 26.3766251
9: -18.9393044, 8.0219088, -18.9092789, 7.9958038, -23.2578888, 23.2681122
10: -16.6577263, 10.9581528, -16.6306515, 10.8564606, -25.2554932, 25.3277130
11: -5.9142761, 16.3178120, -5.8502994, 16.2152195, -17.5015488, 17.5329437
12: -22.5796013, 13.5782347, -22.5385551, 13.4365902, -28.9324265, 29.0355377
13: -33.4191895, 6.7222295, -33.3465347, 6.6929746, -30.1507111, 30.1158600
14: -36.8991966, 8.3456383, -36.8306084, 8.2060347, -43.1593933, 43.2265167
15: -17.1430569, 9.4451981, -17.1304035, 9.3983364, -24.0035477, 24.0263290
16: -19.6598148, 3.7474244, -19.6245537, 3.7302396, -20.9421082, 20.9136200
17: -26.3954029, 7.6044827, -26.3455982, 7.4581347, -33.8535385, 33.9500809
18: -7.5776196, 25.3890152, -7.5524406, 25.3341484, -30.9887466, 31.0135803
19: -0.9635391, 16.0837097, -0.9171686, 16.0846004, -15.5136757, 15.4758282
20: -7.0298815, 12.3117056, -6.9853115, 12.3125944, -18.4224167, 18.4141655
21: -5.4999599, 16.2957039, -5.4539595, 16.2600288, -20.8978119, 20.9040413
22: -2.5947800, 16.8689003, -2.5820618, 16.8645782, -16.5750046, 16.5572243
23: -4.0035009, 17.7118320, -3.9750714, 17.6864586, -18.7184525, 18.6951218
24: -2.7668400, 22.2199078, -2.7358952, 22.1833572, -21.5867462, 21.5895004
25: -5.2597847, 18.3057766, -5.2177420, 18.2836037, -20.9383392, 20.9200096
26: -7.7533436, 24.4765816, -7.6864052, 24.3756580, -30.1787567, 30.2114716
27: -5.9351754, 18.1425037, -5.8942471, 18.1247349, -20.5504951, 20.5245743
28: -2.8837690, 20.5357151, -2.8394933, 20.5386925, -21.5733261, 21.5367699
29: -2.4279711, 17.1714954, -2.3929234, 17.1249847, -15.4810791, 15.4905853
30: -9.8144436, 18.6797142, -9.7693539, 18.6003113, -26.1361084, 26.1841812
31: -5.4545174, 17.6641026, -5.4079208, 17.6629543, -21.5573273, 21.5278740
32: -28.6825695, -1.3763218, -28.6216621, -1.4073033, -20.8567390, 20.8337631
33: -50.8328247, -11.6680784, -50.7321014, -11.7296982, -27.6125412, 27.5854797
34: -45.2240524, -13.7612495, -45.2142563, -13.8019009, -24.3492584, 24.3636627
35: -32.3011055, -2.8986695, -32.2643814, -2.9182642, -23.2203789, 23.1802521
36: -29.4364166, 2.3190861, -29.3691635, 2.3046889, -25.2824860, 25.2174225
37: -46.4113617, -5.4875402, -46.3602753, -5.5035033, -36.0694885, 36.0015411
38: -40.1100159, -2.7690339, -40.0771484, -2.7868905, -32.6870270, 32.6593781
39: -50.2904968, -7.9422245, -50.1519165, -8.0008450, -29.2503738, 29.1920471
40: -47.9962349, -17.6781063, -47.9399529, -17.6759682, -24.9383011, 24.8721542
41: -28.7755604, 0.9046431, -28.7380772, 0.9002528, -25.6758423, 25.6371078
42: -32.4876442, -9.5282669, -32.4820366, -9.5591888, -18.7617874, 18.7713356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=238, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1561

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.4978397, upper bound: 19.6887456
time: 33.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5150131, upper bound: 19.6887452
time: 38.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 75.01 seconds
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 75.01
Output dim: 26, lower bound: -19.4978397, upper bound: 19.6887456
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 75.01
Output dim: 26, lower bound: -19.5150131, upper bound: 19.6887452

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -33.8630104, -0.6319160, -33.8067856, -0.6830208, -24.5907059, 24.5944214
1: -13.1118755, 7.4107428, -13.1037540, 7.3939176, -15.9581642, 15.9531517
2: -12.0992584, 8.5992279, -12.0348406, 8.5653019, -15.3849640, 15.3574142
3: -26.8273678, -2.2454910, -26.7834091, -2.3004875, -19.8975983, 19.9046783
4: -16.7604828, 11.0100107, -16.6425552, 10.9629049, -20.8417625, 20.7735596
5: -21.5447941, 3.2291126, -21.4670830, 3.1891046, -19.1982727, 19.1592789
6: -34.7039032, -7.5877781, -34.6791954, -7.6057291, -22.4744339, 22.4765701
7: -20.8353863, 6.1945758, -20.7564545, 6.1671581, -21.3319016, 21.2783813
8: -30.9950829, 4.9882865, -30.9585876, 4.9673481, -26.3841248, 26.3551636
9: -18.9369907, 8.0189705, -18.9035397, 7.9886103, -23.2512589, 23.2590103
10: -16.6547680, 10.9537258, -16.6233292, 10.8457270, -25.2418518, 25.3159637
11: -5.9111943, 16.3168201, -5.8427792, 16.2128315, -17.4952278, 17.5240402
12: -22.5782909, 13.5711555, -22.5353012, 13.4197063, -28.9144363, 29.0251770
13: -33.4033661, 6.7192893, -33.3080292, 6.6859155, -30.1281891, 30.0747299
14: -36.8902740, 8.3426628, -36.8090668, 8.1986237, -43.1412201, 43.2027740
15: -17.1397114, 9.4382744, -17.1222267, 9.3820305, -23.9859848, 24.0132103
16: -19.6571350, 3.7449379, -19.6179771, 3.7241280, -20.9343643, 20.9048233
17: -26.3914680, 7.6003914, -26.3359146, 7.4479761, -33.8394432, 33.9363060
18: -7.5747452, 25.3759079, -7.5454588, 25.3017750, -30.9495773, 30.9923401
19: -0.9601746, 16.0832043, -0.9089212, 16.0833492, -15.5076962, 15.4659004
20: -7.0242720, 12.3104315, -6.9714785, 12.3095293, -18.4132309, 18.3995819
21: -5.4921379, 16.2945499, -5.4346876, 16.2571735, -20.8874893, 20.8852234
22: -2.5919490, 16.8672543, -2.5751047, 16.8605843, -16.5662956, 16.5485458
23: -4.0012589, 17.7023907, -3.9696102, 17.6630535, -18.6990128, 18.6823921
24: -2.7638540, 22.2174492, -2.7286911, 22.1772766, -21.5755730, 21.5788918
25: -5.2544827, 18.3036690, -5.2047257, 18.2783127, -20.9274864, 20.9056435
26: -7.7503052, 24.4697952, -7.6788883, 24.3590736, -30.1567764, 30.1961670
27: -5.9330769, 18.1370277, -5.8891478, 18.1116486, -20.5348892, 20.5141792
28: -2.8816376, 20.5320129, -2.8342700, 20.5296402, -21.5625305, 21.5276566
29: -2.4254766, 17.1685123, -2.3867874, 17.1176891, -15.4693260, 15.4807854
30: -9.8090496, 18.6759415, -9.7561893, 18.5908546, -26.1207886, 26.1684418
31: -5.4473410, 17.6629181, -5.3903790, 17.6599827, -21.5464554, 21.5125389
32: -28.6801033, -1.3800683, -28.6155930, -1.4163394, -20.8450317, 20.8229485
33: -50.8307877, -11.6827106, -50.7271271, -11.7656574, -27.5768433, 27.5669403
34: -45.2220840, -13.7734804, -45.2093887, -13.8320904, -24.3169861, 24.3471603
35: -32.2979698, -2.8997908, -32.2567940, -2.9209642, -23.2136688, 23.1718292
36: -29.4324970, 2.3175788, -29.3595791, 2.3008351, -25.2744446, 25.2058716
37: -46.4084244, -5.5130610, -46.3531647, -5.5666804, -36.0011368, 35.9678192
38: -40.1058960, -2.7709584, -40.0670929, -2.7915397, -32.6718369, 32.6429596
39: -50.2868309, -7.9439597, -50.1429520, -8.0051479, -29.2415695, 29.1801529
40: -47.9939651, -17.7046528, -47.9343719, -17.7416306, -24.8690796, 24.8405914
41: -28.7735367, 0.8880544, -28.7332840, 0.8591685, -25.6334534, 25.6150970
42: -32.4865417, -9.5472193, -32.4793091, -9.6059771, -18.7155762, 18.7492943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1532

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4978396, upper bound: 19.6392137
time: 29.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.4978396, upper bound: 19.6887456
time: 42.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -33.8645935, -0.6237607, -33.8843651, -0.6498578, -24.6170044, 24.6769791
1: -13.1134624, 7.4116783, -13.1127415, 7.4301357, -16.0048103, 15.9619331
2: -12.1029282, 8.5999279, -12.0504818, 8.6020164, -15.4276199, 15.3739319
3: -26.8327675, -2.2453017, -26.8008461, -2.2491341, -19.9578247, 19.9215126
4: -16.7688065, 11.0105724, -16.6682529, 11.0273781, -20.9166336, 20.7965126
5: -21.5514374, 3.2297039, -21.4883823, 3.2465644, -19.2628174, 19.1772804
6: -34.7043076, -7.5830488, -34.7253571, -7.5728159, -22.5116653, 22.5321198
7: -20.8430328, 6.1956406, -20.7805405, 6.2355757, -21.4107285, 21.3009720
8: -31.0001507, 4.9892993, -30.9746017, 5.0082946, -26.4383392, 26.3734436
9: -18.9384575, 8.0192394, -18.9413471, 8.0016518, -23.2804871, 23.2738419
10: -16.6563492, 10.9566727, -16.6558094, 10.8794346, -25.2755432, 25.3507233
11: -5.9118719, 16.3175030, -5.8600397, 16.2283707, -17.5108948, 17.5517082
12: -22.5789566, 13.5754147, -22.5699768, 13.4414253, -28.9391632, 29.0680008
13: -33.4150162, 6.7202516, -33.3481979, 6.7700033, -30.2268372, 30.1102448
14: -36.8900452, 8.3441048, -36.8528709, 8.2281313, -43.1649933, 43.2596436
15: -17.1421280, 9.4420414, -17.1730499, 9.4062719, -24.0087814, 24.0676117
16: -19.6576424, 3.7459378, -19.6568642, 3.7452338, -20.9710846, 20.9370804
17: -26.3883038, 7.6021900, -26.3538437, 7.4641390, -33.8524437, 33.9560318
18: -7.5760031, 25.3854713, -7.6190605, 25.3326836, -30.9841156, 31.0783920
19: -0.9613848, 16.0834866, -0.9367800, 16.0923920, -15.5159893, 15.5016594
20: -7.0280027, 12.3105793, -6.9997244, 12.3449268, -18.4518013, 18.4311028
21: -5.4967709, 16.2953148, -5.4696007, 16.2913265, -20.9230881, 20.9188309
22: -2.5921016, 16.8679657, -2.6076307, 16.8787575, -16.5754204, 16.6032410
23: -4.0020676, 17.7096233, -4.0392370, 17.6845360, -18.7123795, 18.7548027
24: -2.7640953, 22.2177258, -2.7642002, 22.1857452, -21.5844727, 21.6232872
25: -5.2556500, 18.3048725, -5.2322645, 18.3157253, -20.9637909, 20.9364204
26: -7.7521410, 24.4729557, -7.7663498, 24.3762589, -30.1727371, 30.2946625
27: -5.9335885, 18.1405621, -5.9343424, 18.1252480, -20.5484848, 20.5698509
28: -2.8828850, 20.5326462, -2.8951807, 20.5375252, -21.5754471, 21.5847931
29: -2.4264972, 17.1698723, -2.4390011, 17.1249390, -15.4770126, 15.5414276
30: -9.8113899, 18.6785374, -9.7823601, 18.6196289, -26.1454620, 26.2004471
31: -5.4495754, 17.6636658, -5.4202909, 17.6944427, -21.5696030, 21.5550995
32: -28.6812305, -1.3798876, -28.6459579, -1.3953962, -20.8702850, 20.8519554
33: -50.8318596, -11.6719913, -50.8149796, -11.7224588, -27.6126900, 27.6645737
34: -45.2228088, -13.7645607, -45.2819672, -13.8034706, -24.3430481, 24.4335670
35: -32.2994080, -2.9000034, -32.2796822, -2.9150949, -23.2211380, 23.1989594
36: -29.4337311, 2.3185444, -29.3905849, 2.3219914, -25.2940140, 25.2445831
37: -46.4092903, -5.4934578, -46.4896126, -5.5068913, -36.0496674, 36.1307755
38: -40.1080360, -2.7710071, -40.0949173, -2.7743845, -32.6888504, 32.6818466
39: -50.2873535, -7.9430027, -50.1557236, -7.9844999, -29.2626495, 29.1957245
40: -47.9953537, -17.6848488, -48.0516434, -17.6810036, -24.9193726, 24.9752121
41: -28.7736969, 0.9005857, -28.8222694, 0.9031358, -25.6710052, 25.7150803
42: -32.4870644, -9.5333595, -32.5706253, -9.5575771, -18.7547112, 18.8516922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1532

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5150131, upper bound: 19.6392142
time: 42.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -19.5150131, upper bound: 19.6887456
time: 31.64 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 75.98 seconds
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 75.98
Output dim: 26, lower bound: -19.4978396, upper bound: 19.6392137
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 75.98
Output dim: 26, lower bound: -19.4978396, upper bound: 19.6887456
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 75.98
Output dim: 26, lower bound: -19.5150131, upper bound: 19.6392142
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 75.98
Output dim: 26, lower bound: -19.5150131, upper bound: 19.6887456

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -33.8980598, -0.6219587, -33.8067856, -0.6830208, -24.6216812, 24.6055756
1: -13.1171408, 7.4178114, -13.1037540, 7.3939176, -15.9629784, 15.9600983
2: -12.1379776, 8.6078711, -12.0348406, 8.5653019, -15.4227295, 15.3669205
3: -26.8550549, -2.2293558, -26.7834091, -2.3004875, -19.9281006, 19.9194565
4: -16.8257942, 11.0206232, -16.6425552, 10.9629049, -20.9119530, 20.7836304
5: -21.5920963, 3.2432208, -21.4670830, 3.1891046, -19.2496796, 19.1739426
6: -34.7178726, -7.5794296, -34.6791954, -7.6057291, -22.5012436, 22.4846954
7: -20.8834038, 6.2064333, -20.7564545, 6.1671581, -21.3828812, 21.2894592
8: -31.0143509, 4.9974642, -30.9585876, 4.9673481, -26.3954239, 26.3667145
9: -18.9582500, 8.0318213, -18.9035397, 7.9886103, -23.2701035, 23.2701035
10: -16.6698475, 11.0025263, -16.6233292, 10.8457270, -25.2561493, 25.3628922
11: -5.9223456, 16.3770866, -5.8427792, 16.2128315, -17.5057716, 17.5822868
12: -22.5915909, 13.6690102, -22.5353012, 13.4197063, -28.9303360, 29.1304092
13: -33.4401207, 6.7405672, -33.3080292, 6.6859155, -30.1684875, 30.0983200
14: -36.9305649, 8.4375458, -36.8090668, 8.1986237, -43.1796417, 43.2987061
15: -17.1535454, 9.4493780, -17.1222267, 9.3820305, -23.9987259, 24.0289001
16: -19.6731606, 3.7538037, -19.6179771, 3.7241280, -20.9515228, 20.9148865
17: -26.4116592, 7.7121162, -26.3359146, 7.4479761, -33.8596344, 34.0480309
18: -7.6011114, 25.4038906, -7.5454588, 25.3017750, -30.9760818, 31.0159683
19: -0.9832401, 16.0849400, -0.9089212, 16.0833492, -15.5320263, 15.4671516
20: -7.0447030, 12.3176928, -6.9714785, 12.3095293, -18.4406586, 18.4027519
21: -5.5109634, 16.3150711, -5.4346876, 16.2571735, -20.9127579, 20.9022064
22: -2.6083860, 16.8741913, -2.5751047, 16.8605843, -16.5787544, 16.5604553
23: -4.0163298, 17.7175884, -3.9696102, 17.6630535, -18.7133484, 18.6977196
24: -2.7846718, 22.2368622, -2.7286911, 22.1772766, -21.5961342, 21.6024246
25: -5.2717614, 18.3218613, -5.2047257, 18.2783127, -20.9444389, 20.9243164
26: -7.7757912, 24.5326309, -7.6788883, 24.3590736, -30.1821747, 30.2618790
27: -5.9605713, 18.1472111, -5.8891478, 18.1116486, -20.5627441, 20.5279770
28: -2.9010191, 20.5378876, -2.8342700, 20.5296402, -21.5848083, 21.5338326
29: -2.4394104, 17.2034798, -2.3867874, 17.1176891, -15.4821854, 15.5177803
30: -9.8211021, 18.7244720, -9.7561893, 18.5908546, -26.1342621, 26.2096405
31: -5.4760275, 17.6651993, -5.3903790, 17.6599827, -21.5761566, 21.5148048
32: -28.7180634, -1.3673167, -28.6155930, -1.4163394, -20.8881149, 20.8348846
33: -50.8896790, -11.6707249, -50.7271271, -11.7656574, -27.6285019, 27.5798416
34: -45.2368889, -13.7620726, -45.2093887, -13.8320904, -24.3328323, 24.3580856
35: -32.3240814, -2.8930609, -32.2567940, -2.9209642, -23.2408524, 23.1791840
36: -29.4691620, 2.3246408, -29.3595791, 2.3008351, -25.3118973, 25.2129974
37: -46.4392815, -5.5034404, -46.3531647, -5.5666804, -36.0329361, 35.9829559
38: -40.1320000, -2.7610464, -40.0670929, -2.7915397, -32.6985855, 32.6543503
39: -50.3681641, -7.9348297, -50.1429520, -8.0051479, -29.3164520, 29.1919861
40: -48.0274582, -17.6991348, -47.9343719, -17.7416306, -24.9047699, 24.8469772
41: -28.7938118, 0.8974748, -28.7332840, 0.8591685, -25.6580734, 25.6246109
42: -32.4942551, -9.5252991, -32.4793091, -9.6059771, -18.7211227, 18.7776871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=237, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4802066, upper bound: 19.6799347
time: 27.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4892471, upper bound: 19.6799342
time: 39.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -33.8996696, -0.6137910, -33.8843651, -0.6498578, -24.6479721, 24.6881332
1: -13.1187000, 7.4187450, -13.1127415, 7.4301357, -16.0096169, 15.9688759
2: -12.1416817, 8.6086006, -12.0504818, 8.6020164, -15.4654007, 15.3834190
3: -26.8604164, -2.2291489, -26.8008461, -2.2491341, -19.9883270, 19.9362793
4: -16.8341007, 11.0212135, -16.6682529, 11.0273781, -20.9868317, 20.8065872
5: -21.5987797, 3.2437997, -21.4883823, 3.2465644, -19.3142395, 19.1919441
6: -34.7182312, -7.5746717, -34.7253571, -7.5728159, -22.5384903, 22.5402489
7: -20.8910580, 6.2074909, -20.7805405, 6.2355757, -21.4616852, 21.3120346
8: -31.0194263, 4.9985094, -30.9746017, 5.0082946, -26.4496536, 26.3850174
9: -18.9597054, 8.0320673, -18.9413471, 8.0016518, -23.2993469, 23.2849426
10: -16.6714039, 11.0055199, -16.6558094, 10.8794346, -25.2898407, 25.3976440
11: -5.9230237, 16.3777561, -5.8600397, 16.2283707, -17.5214233, 17.6099586
12: -22.5922165, 13.6732445, -22.5699768, 13.4414253, -28.9550781, 29.1732635
13: -33.4517097, 6.7414889, -33.3481979, 6.7700033, -30.2671432, 30.1338654
14: -36.9303131, 8.4390831, -36.8528709, 8.2281313, -43.2034454, 43.3555603
15: -17.1559601, 9.4531555, -17.1730499, 9.4062719, -24.0215454, 24.0833130
16: -19.6736717, 3.7548144, -19.6568642, 3.7452338, -20.9882507, 20.9472046
17: -26.4085388, 7.7139826, -26.3538437, 7.4641390, -33.8726768, 34.0678253
18: -7.6023493, 25.4135056, -7.6190605, 25.3326836, -31.0105896, 31.1020355
19: -0.9844179, 16.0852184, -0.9367800, 16.0923920, -15.5403576, 15.5029049
20: -7.0484533, 12.3178463, -6.9997244, 12.3449268, -18.4792671, 18.4342690
21: -5.5156097, 16.3158417, -5.4696007, 16.2913265, -20.9483795, 20.9358177
22: -2.6085644, 16.8748779, -2.6076307, 16.8787575, -16.5878944, 16.6151428
23: -4.0171208, 17.7248173, -4.0392370, 17.6845360, -18.7267227, 18.7701492
24: -2.7849522, 22.2371025, -2.7642002, 22.1857452, -21.6050262, 21.6468391
25: -5.2729154, 18.3230667, -5.2322645, 18.3157253, -20.9807358, 20.9550743
26: -7.7776222, 24.5358238, -7.7663498, 24.3762589, -30.1981659, 30.3603973
27: -5.9610901, 18.1507263, -5.9343424, 18.1252480, -20.5763474, 20.5836449
28: -2.9022498, 20.5385361, -2.8951807, 20.5375252, -21.5977325, 21.5909576
29: -2.4404297, 17.2048264, -2.4390011, 17.1249390, -15.4898834, 15.5784225
30: -9.8234625, 18.7271194, -9.7823601, 18.6196289, -26.1589737, 26.2416534
31: -5.4783096, 17.6659412, -5.4202909, 17.6944427, -21.5993195, 21.5573616
32: -28.7191887, -1.3671079, -28.6459579, -1.3953962, -20.9133682, 20.8638763
33: -50.8908463, -11.6599817, -50.8149796, -11.7224588, -27.6643486, 27.6774292
34: -45.2376175, -13.7531815, -45.2819672, -13.8034706, -24.3588867, 24.4444504
35: -32.3254623, -2.8932848, -32.2796822, -2.9150949, -23.2483673, 23.2062988
36: -29.4703903, 2.3256865, -29.3905849, 2.3219914, -25.3315048, 25.2516785
37: -46.4401474, -5.4838834, -46.4896126, -5.5068913, -36.0814972, 36.1459198
38: -40.1340485, -2.7610693, -40.0949173, -2.7743845, -32.7155838, 32.6932373
39: -50.3687210, -7.9339361, -50.1557236, -7.9844999, -29.3375626, 29.2075577
40: -48.0288963, -17.6793365, -48.0516434, -17.6810036, -24.9550400, 24.9815979
41: -28.7939796, 0.9099550, -28.8222694, 0.9031358, -25.6955872, 25.7245941
42: -32.4947815, -9.5114384, -32.5706253, -9.5575771, -18.7602501, 18.8800812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=237, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=31, inp2_unstable=31, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1532

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.4973963, upper bound: 19.6799347
time: 44.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -19.5063698, upper bound: 19.6799347
time: 36.59 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 83.57 seconds
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 83.57
Output dim: 26, lower bound: -19.4802066, upper bound: 19.6799347
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 83.57
Output dim: 26, lower bound: -19.4892471, upper bound: 19.6799342
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 83.57
Output dim: 26, lower bound: -19.4973963, upper bound: 19.6799347
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 83.57
Output dim: 26, lower bound: -19.5063698, upper bound: 19.6799347

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 47.83 + 1418.33 = 1466.17 seconds
